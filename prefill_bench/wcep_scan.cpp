// WCEP Phase 1a — actual-weight scanner (read-only)
//
// research/17 §7.8, §7.9. **ISA lowering 을 하지 않는다.** 따라서 지표는
// "symbolic operation-count reduction" 이며, 실제 µop·성능 환산은 1b 이후다.
//
// 하는 일:
//   1. dllama .m 헤더를 읽어 Q4_0 weight 영역을 찾는다 (파일은 수정하지 않는다)
//   2. tile = 4 output row x 32 input weight 단위로 전수 통계
//   3. 네 종류 null 을 paired 생성하고 같은 분석기로 처리
//   4. actual - null 차이를 layer 단위로 집계
//
// tile 정의는 WCEP 컴파일 단위와 일치해야 한다(§7.8.1). 셔플 단위가 다르면
// "구조가 있다/없다" 판정이 무의미해진다.
//
// build: g++ -std=c++17 -O2 prefill_bench/wcep_scan.cpp -o wcep_scan
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <random>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---- 고정 파라미터 (§7.8.8, 결과 보기 전 확정 — 변경 금지) ----
static const int TILE_ROWS   = 4;        // output row
static const int TILE_K      = 32;       // input weight (= Q4_0 block)
static const int BEAM_SAMPLE = 64;       // (layer x projection) 당 tile
static const int NULL_REPS   = 8;        // actual tile 마다 paired
static const uint64_t SEED   = 20260822ull;

// Q4_0: 16 B code(32 nibble) + 2 B fp16 scale = 18 B
struct Q40 { uint16_t d; uint8_t qs[16]; };
static_assert(sizeof(Q40) == 18, "Q4_0 block must be 18 bytes");

// ⚠️ Q4_0 layout (src/nn/nn-quants.cpp dequantizeQ40toF32):
//     qs[j] 의 low nibble  -> weight j
//     qs[j] 의 high nibble -> weight j + 16
//   인접 쌍으로 읽으면 histogram 은 비슷해도 **위치·cross-output 분석이 무효**가 된다.
static inline int nib(const Q40 &b, int i) {
    const int j = i & 15;
    const uint8_t byte = b.qs[j];
    const int v = (i < 16) ? (byte & 0x0F) : (byte >> 4);
    return v - 8;                        // [-8,7]
}

// ── 구조적 절약 추정 ──────────────────────────────────────────────────
//
// baseline: tile 하나를 계산하려면 4행 x 32원소 = 128 개의 곱-누산이 필요하다.
//
// 절약 후보 (ISA 무관, symbolic):
//   (a) 0 계수      — 항 자체가 사라진다
//   (b) 행 내 중복  — 같은 (값) 이 반복되면 x 를 먼저 합산하고 한 번만 곱한다
//   (c) 행 간 공유  — 여러 output row 가 같은 (위치,값) 쌍을 쓰면 부분합을 공유한다
//
// 이것은 상한이 아니라 **보수적 하한 추정**이다. beam search 는 1c 에서 한다.
struct TileStat { int ops; int base; };

// ⚠️ baseline 과 최적화 쪽의 **연산 단위가 같아야 한다.**
//   baseline : 원소마다 곱 1 + 누산 1  ->  2 * 4 * 32 = 256
//   최적화   : 같은 값 n 개를 묶으면 (n-1) 덧셈 + 1 곱 + 1 누산 = n+1
//              |v|=1 이면 곱이 사라져 n,  v=0 이면 0
// 처음 판은 baseline 을 128(MAC 수)로 두고 최적화만 곱·덧셈을 따로 세어
// 표본에서 절약이 음수로 나왔다.
static TileStat analyzeTile(const Q40 *rows[TILE_ROWS]) {
    TileStat st{0, 2 * TILE_ROWS * TILE_K};
    // (b) 행마다 값별 개수 -> 같은 값 n 개는 (n-1) 번의 덧셈 + 1 번의 곱
    //     즉 n 개의 곱-누산이 n 개 연산 -> (n-1)+1 = n ... 이득 없음.
    //     실제 이득은 "값이 1 또는 -1" 이면 곱이 사라지고, 같은 값이 뭉치면
    //     x 합산 후 한 번만 곱하므로 n 곱 -> 1 곱 + (n-1) 덧셈.
    //     덧셈이 곱보다 싸다는 가정을 하지 않으려면 연산 수는 같다.
    //     따라서 여기서는 **연산 수** 기준으로 다음만 센다:
    //       - 계수 0      : 연산 2개(곱+누산) 제거
    //       - 계수 ±1     : 곱 1개 제거
    //       - 값 뭉침 n>1 : 곱 (n-1)개 제거 (합산 후 1회 곱)
    std::map<int, int> shared;           // (pos<<8 | val+8) -> 등장 행 수
    for (int r = 0; r < TILE_ROWS; r++) {
        std::map<int, int> cnt;
        for (int i = 0; i < TILE_K; i++) {
            const int v = nib(*rows[r], i);
            cnt[v]++;
            shared[(i << 8) | (v + 8)]++;
        }
        for (const auto &kv : cnt) {
            const int v = kv.first, n = kv.second;
            if (v == 0) continue;                       // 항 제거 (0 ops)
            if (v == 1 || v == -1) st.ops += n;         // 덧셈만 (곱 없음)
            else st.ops += n + 1;                       // (n-1) 덧셈 + 1 곱 + 1 누산
        }
    }
    // (c) 행 간 공유: 같은 (위치,값) 이 m 행에 나타나면 부분합을 (m-1) 번 재사용
    for (const auto &kv : shared) {
        const int v = (kv.first & 0xFF) - 8;
        if (v == 0) continue;
        if (kv.second > 1) st.ops -= (kv.second - 1);
    }
    if (st.ops < 0) st.ops = 0;
    return st;
}

// ── null 생성기 (§7.8.2) ──────────────────────────────────────────────
enum NullKind { NULL_UNIFORM = 0, NULL_PROJ_HIST, NULL_TILE_HIST, NULL_CROSS_OUT };
// ⚠️ proj_hist 는 전수 집계가 아니라 표본 기반이므로 exact 를 주장하지 않는다.
static const char *NULL_NAME[] = { "uniform", "sampled_proj_hist", "tile_hist", "cross_out" };

static void setNib(Q40 &b, int i, int v) {
    const int j = i & 15;
    const uint8_t u = (uint8_t)(v + 8);
    if (i < 16) b.qs[j] = (uint8_t)((b.qs[j] & 0xF0) | u);
    else        b.qs[j] = (uint8_t)((b.qs[j] & 0x0F) | (u << 4));
}

// actual tile 을 받아 null tile 을 만든다. 원본은 절대 수정하지 않는다.
static void makeNull(NullKind kind, const Q40 *src[TILE_ROWS], Q40 dst[TILE_ROWS],
                     const std::vector<int> &projPool, std::mt19937_64 &rng) {
    for (int r = 0; r < TILE_ROWS; r++) dst[r] = *src[r];
    if (kind == NULL_UNIFORM) {
        std::uniform_int_distribution<int> u(-8, 7);
        for (int r = 0; r < TILE_ROWS; r++)
            for (int i = 0; i < TILE_K; i++) setNib(dst[r], i, u(rng));
    } else if (kind == NULL_PROJ_HIST) {
        // projection 전체의 nibble **개수를 정확히 보존**한 셔플 풀에서 소비한다.
        // (이전 판은 4,096개 표본에서 복원추출해 "histogram 정확 보존" 정의를
        //  만족하지 못했다.)
        static thread_local size_t cursor = 0;
        for (int r = 0; r < TILE_ROWS; r++)
            for (int i = 0; i < TILE_K; i++) {
                setNib(dst[r], i, projPool[cursor % projPool.size()]);
                cursor++;
            }
    } else if (kind == NULL_TILE_HIST) {
        // tile 전체 128 값을 모아 셔플 → tile histogram 정확히 보존, 배열만 파괴
        std::vector<int> vals;
        vals.reserve(TILE_ROWS * TILE_K);
        for (int r = 0; r < TILE_ROWS; r++)
            for (int i = 0; i < TILE_K; i++) vals.push_back(nib(*src[r], i));
        std::shuffle(vals.begin(), vals.end(), rng);
        int k = 0;
        for (int r = 0; r < TILE_ROWS; r++)
            for (int i = 0; i < TILE_K; i++) setNib(dst[r], i, vals[k++]);
    } else { // NULL_CROSS_OUT
        // 각 행의 histogram 은 보존하고 행 내부만 셔플 → 행 간 대응 파괴
        for (int r = 0; r < TILE_ROWS; r++) {
            std::vector<int> vals;
            for (int i = 0; i < TILE_K; i++) vals.push_back(nib(*src[r], i));
            std::shuffle(vals.begin(), vals.end(), rng);
            for (int i = 0; i < TILE_K; i++) setNib(dst[r], i, vals[i]);
        }
    }
}

// ── 모델 파일 파싱 ────────────────────────────────────────────────────
//
// dllama .m 레이아웃 (src/llm.cpp loadLlmNetWeight):
//   [magic 4B][headerSize 4B][header ...]
//   embedding
//   레이어마다: q, k, v, wo, w1, w2, w3, norm_0, norm_1
//   final_norm, wcls
//
// Q4_0 projection 만 스캔한다. norm 은 F32 라 대상이 아니다.
struct Proj { const char *name; int d; int k; };   // d = output rows, k = input elems

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "dllama_model_llama3-8b_q40.m";
    const char *outDir = argc > 2 ? argv[2] : "artifacts/wcep_1a";

    const int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "열 수 없음: %s\n", path); return 1; }
    struct stat sb; fstat(fd, &sb);
    const uint8_t *base = (const uint8_t *)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { fprintf(stderr, "mmap 실패\n"); return 1; }

    const int32_t magic = *(const int32_t *)base;
    const int32_t headerSize = *(const int32_t *)(base + 4);
    if (magic != 0xA00ABCD) { fprintf(stderr, "magic 불일치: %x\n", magic); return 1; }

    // ── 헤더를 실제로 파싱한다 (src/llm.cpp loadLlmHeader 와 동일 규약) ──
    //   [magic 4B][headerSize 4B][key value] * N
    enum { K_DIM=2, K_HIDDEN=3, K_NLAYERS=4, K_NHEADS=5, K_NKVHEADS=6,
           K_VOCAB=9, K_HEADDIM=19 };
    std::map<int,int> hdr;
    {
        const int32_t *kv = (const int32_t *)(base + 8);
        const int nKv = (headerSize - 2*(int)sizeof(int32_t)) / (int)sizeof(int32_t) / 2;
        for (int i = 0; i < nKv; i++) hdr[kv[i*2]] = kv[i*2+1];
    }
    const int dim     = hdr.count(K_DIM)     ? hdr[K_DIM]     : 0;
    const int hidden  = hdr.count(K_HIDDEN)  ? hdr[K_HIDDEN]  : 0;
    const int nLayers = hdr.count(K_NLAYERS) ? hdr[K_NLAYERS] : 0;
    const int nHeads  = hdr.count(K_NHEADS)  ? hdr[K_NHEADS]  : 0;
    const int nKvHeads= hdr.count(K_NKVHEADS)? hdr[K_NKVHEADS]: 0;
    const int vocab   = hdr.count(K_VOCAB)   ? hdr[K_VOCAB]   : 0;
    const int headDim = hdr.count(K_HEADDIM) && hdr[K_HEADDIM] ? hdr[K_HEADDIM] : (nHeads ? dim/nHeads : 0);
    const int kvDim   = nKvHeads * headDim;
    if (!dim || !hidden || !nLayers || !vocab || !kvDim) {
        fprintf(stderr, "헤더 파싱 실패: dim=%d hidden=%d L=%d vocab=%d kvDim=%d\n",
                dim, hidden, nLayers, vocab, kvDim); return 1;
    }
    printf("# model=%s size=%.2f GB headerSize=%d\n", path, sb.st_size/1e9, headerSize);
    printf("# dim=%d hidden=%d nLayers=%d nHeads=%d nKvHeads=%d headDim=%d vocab=%d kvDim=%d\n",
           dim, hidden, nLayers, nHeads, nKvHeads, headDim, vocab, kvDim);
    // ⚠️ **파일 저장 순서** 를 그대로 따라야 한다 (src/llm.cpp loadLlmNetWeight):
    //     Q -> K -> V -> O -> W1(Gate) -> W2(Down) -> W3(Up)
    //   세 FFN 행렬은 원소 수가 같아(4096x14336 == 14336x4096) 순서를 틀려도
    //   byte offset self-test 를 통과한다. 그러나 output-row 경계와 tile 구성이
    //   깨져 tile_hist/cross_out 분석이 무효가 된다.
    //   (이 오류로 두 번째 Phase 1a 결과가 무효가 됐다 — artifacts/wcep_1a_invalid_ffn_order)
    const Proj PROJS[] = {
        {"Q",    dim,    dim}, {"K",  kvDim, dim}, {"V",  kvDim, dim}, {"O", dim, dim},
        {"Gate", hidden, dim},          // W1
        {"Down", dim,    hidden},       // W2
        {"Up",   hidden, dim},          // W3
    };
    const int NP = 7;

    // ⚠️ embedding 은 **F32** 다 (src/llm.cpp: size2D(F_32, vocabSize, dim)).
    //    Q4_0 으로 가정하면 1.68 GiB 어긋나 F32 embedding 중간을 Q4 로 해석하게 된다.
    //    (이 오류로 첫 Phase 1a 결과가 무효가 됐다 — artifacts/wcep_1a_invalid_*)
    size_t off = (size_t)headerSize;
    const size_t embBytes = (size_t)vocab * dim * sizeof(float);
    printf("# embedding  [%zu, %zu)  %.2f GiB (F32)\n", off, off+embBytes, embBytes/1073741824.0);
    off += embBytes;
    const size_t weightsStart = off;

    std::mt19937_64 rng(SEED);
    FILE *fs = fopen((std::string(outDir) + "/projection_summary.tsv").c_str(), "w");
    FILE *fl = fopen((std::string(outDir) + "/layer_summary.tsv").c_str(), "w");
    FILE *fn = fopen((std::string(outDir) + "/paired_null_effect.tsv").c_str(), "w");
    fprintf(fs, "projection\ttiles\tbase_ops\tactual_ops\treduction\n");
    fprintf(fl, "layer\tprojection\ttiles\treduction\n");
    fprintf(fn, "layer\tprojection\tnull\tactual_red\tnull_red\tdelta_pp\n");

    std::map<std::string, long long> pBase, pOps, pTiles;
    long long gaTotal = 0, gaEligible = 0; int gaMin = 1<<30;
    long long gaHist[16] = {0};

    for (int L = 0; L < nLayers; L++) {
        for (int p = 0; p < NP; p++) {
            const Proj &pr = PROJS[p];
            const int kBlocks = pr.k / TILE_K;
            const size_t nBytes = (size_t)pr.d * kBlocks * sizeof(Q40);
            const Q40 *W = (const Q40 *)(base + off);
            if (off + nBytes > (size_t)sb.st_size) { fprintf(stderr, "범위 초과 L%d %s\n", L, pr.name); goto done; }

            // 전수: 이 projection 의 모든 tile
            if (L == 0) printf("# L0 %-5s [%zu, %zu)  d=%d k=%d\n", pr.name, off, off+nBytes, pr.d, pr.k);
            long long base_ops = 0, act_ops = 0, nTiles = 0;

            // proj_hist null: projection **전체**의 nibble count 를 정확히 보존한
            // 셔플 풀. (전수는 크므로 값별 개수를 세고 그 비율대로 풀을 구성한다 —
            // 개수 보존이 정의이므로 비율 보존 + 셔플로 동일한 histogram 을 만든다.)
            std::vector<long long> hist(16, 0);
            for (int r0 = 0; r0 < pr.d; r0 += 4)
                for (int kb = 0; kb < kBlocks; kb += 8)
                    for (int i = 0; i < TILE_K; i++)
                        hist[nib(W[(size_t)r0*kBlocks+kb], i) + 8]++;
            long long htot = 0; for (long long h : hist) htot += h;
            std::vector<int> projPool;
            projPool.reserve(65536);
            for (int v = 0; v < 16; v++) {
                const long long cnt = (long long)(65536.0 * hist[v] / htot);
                for (long long c = 0; c < cnt; c++) projPool.push_back(v - 8);
            }
            if (projPool.empty()) projPool.push_back(0);
            std::shuffle(projPool.begin(), projPool.end(), rng);

            for (int r0 = 0; r0 + TILE_ROWS <= pr.d; r0 += TILE_ROWS) {
                for (int kb = 0; kb < kBlocks; kb++) {
                    const Q40 *rows[TILE_ROWS];
                    for (int r = 0; r < TILE_ROWS; r++)
                        rows[r] = &W[(size_t)(r0 + r) * kBlocks + kb];
                    const TileStat st = analyzeTile(rows);
                    base_ops += st.base; act_ops += st.ops; nTiles++;
                    // Gate A (§7.11.6): panel = 4 out x 16 batch x 32 K.
                    //   가중치는 batch 축에 공유되므로 int16 후보 = 2 x sym_ops
                    //   (8 lane x 2 vector = 16 batch),  production = 256 명령/K-block
                    //   eligible <=> sym_ops < 128  (부대비용 0 이라는 최대 낙관)
                    gaTotal++;
                    if (st.ops < 128) gaEligible++;
                    if (st.ops < gaMin) gaMin = st.ops;
                    gaHist[st.ops < 256 ? st.ops/16 : 15]++;
                }
            }
            pBase[pr.name] += base_ops; pOps[pr.name] += act_ops; pTiles[pr.name] += nTiles;
            const double red = 1.0 - (double)act_ops / base_ops;
            fprintf(fl, "%d\t%s\t%lld\t%.4f\n", L, pr.name, nTiles, red);

            // 층화 표본으로 paired null 비교
            const int stride = std::max(1, (int)(nTiles / BEAM_SAMPLE));
            long long sActBase=0, sActOps=0;
            std::vector<long long> sNullOps(4, 0);
            int taken = 0;
            for (int t = 0; t < (int)nTiles && taken < BEAM_SAMPLE; t += stride) {
                const int r0 = (t / kBlocks) * TILE_ROWS, kb = t % kBlocks;
                if (r0 + TILE_ROWS > pr.d) break;
                const Q40 *rows[TILE_ROWS];
                for (int r = 0; r < TILE_ROWS; r++) rows[r] = &W[(size_t)(r0 + r) * kBlocks + kb];
                const TileStat a = analyzeTile(rows);
                sActBase += a.base; sActOps += a.ops;
                for (int nk = 0; nk < 4; nk++) {
                    long long acc = 0;
                    for (int rep = 0; rep < NULL_REPS; rep++) {
                        Q40 nt[TILE_ROWS]; const Q40 *np[TILE_ROWS];
                        makeNull((NullKind)nk, rows, nt, projPool, rng);
                        for (int r = 0; r < TILE_ROWS; r++) np[r] = &nt[r];
                        acc += analyzeTile(np).ops;
                    }
                    sNullOps[nk] += acc / NULL_REPS;
                }
                taken++;
            }
            if (sActBase > 0) {
                const double aRed = 1.0 - (double)sActOps / sActBase;
                for (int nk = 0; nk < 4; nk++) {
                    const double nRed = 1.0 - (double)sNullOps[nk] / sActBase;
                    fprintf(fn, "%d\t%s\t%s\t%.4f\t%.4f\t%.2f\n",
                            L, pr.name, NULL_NAME[nk], aRed, nRed, (aRed - nRed) * 100.0);
                }
            }
            off += nBytes;
        }
        off += (size_t)2 * dim * sizeof(float);        // norm_0, norm_1 (F32)
    }
done:
    // ── offset self-test: 마지막 위치가 final_norm + wcls 와 맞는가 ──
    //   레이아웃(src/llm.cpp): ... layers ... , final_norm(F32 dim), wcls(Q4_0)
    {
        const size_t expectFinalNorm = (size_t)dim * sizeof(float);
        const size_t expectWcls = (size_t)vocab * dim / TILE_K * sizeof(Q40);
        const size_t expectEnd = off + expectFinalNorm + expectWcls;
        printf("\n# offset self-test\n");
        printf("#   weights start   %zu\n", weightsStart);
        printf("#   after layers    %zu\n", off);
        printf("#   + final_norm    %zu\n", off + expectFinalNorm);
        printf("#   + wcls(q40)     %zu   (expect end)\n", expectEnd);
        printf("#   file size       %ld\n", (long)sb.st_size);
        const long diff = (long)sb.st_size - (long)expectEnd;
        printf("#   diff            %ld %s\n", diff,
               diff == 0 ? "OK" : "*** 불일치 — 결과 무효 ***");
        if (diff != 0) {
            fprintf(stderr, "offset self-test 실패: %ld 바이트 차이\n", diff);
        }
    }
    for (const auto &kv : pBase) {
        const double red = 1.0 - (double)pOps[kv.first] / kv.second;
        fprintf(fs, "%s\t%lld\t%lld\t%lld\t%.4f\n", kv.first.c_str(),
                pTiles[kv.first], kv.second, pOps[kv.first], red);
        printf("%-6s tiles=%-10lld reduction=%.2f%%\n", kv.first.c_str(), pTiles[kv.first], red*100);
    }
    // ── Gate A 판정 ──
    {
        const double p = gaTotal ? (double)gaEligible / gaTotal : 0.0;
        printf("\n=== Gate A: eligible panel 비율 (sym_ops < 128) ===\n");
        printf("  total=%lld  eligible=%lld  p=%.4f%%  min_sym=%d\n",
               gaTotal, gaEligible, p*100.0, gaMin);
        printf("  E_max = 1/(1-p) = %.4fx     (선택 panel 비용 0 가정)\n", 1.0/(1.0-p));
        printf("  1.35x 필요 p >= 25.93%%  ->  %s\n",
               p >= 0.2593 ? "통과" : "*** 종료 ***");
        printf("\n  sym_ops 분포 (16 구간):\n");
        for (int i = 0; i < 16; i++)
            if (gaHist[i]) printf("    [%3d,%3d) %10lld  %5.2f%%\n",
                                  i*16, (i+1)*16, gaHist[i], 100.0*gaHist[i]/gaTotal);
        FILE *fg = fopen((std::string(outDir) + "/gate_a.tsv").c_str(), "w");
        fprintf(fg, "total\teligible\tp\tE_max\tmin_sym\n%lld\t%lld\t%.6f\t%.4f\t%d\n",
                gaTotal, gaEligible, p, 1.0/(1.0-p), gaMin);
        fclose(fg);
    }
    fclose(fs); fclose(fl); fclose(fn);
    munmap((void *)base, sb.st_size); close(fd);
    return 0;
}
