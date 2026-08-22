#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-cpu-ops.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"
void reportOpProfile();
#include <climits>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

struct CbRequestState {
    std::vector<int> tokens;
    NnUint nInputTokens;
    NnUint pos;
    NnUint maxPos;
    int token;
    bool finished;
    std::string generatedText;
    NnUint generatedTokens;
};

static std::vector<std::string> loadPromptsFromFile(const char *path) {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error(std::string("Cannot open prompts file: ") + path);
    std::vector<std::string> prompts;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty())
            continue;
        prompts.push_back(line);
    }
    if (prompts.empty())
        throw std::runtime_error("prompts-file is empty");
    return prompts;
}

static bool fileExists(const std::string &path) {
    std::ifstream ifs(path.c_str(), std::ios::binary);
    return ifs.good();
}

static bool ensureParentDirectoryForFileLocal(const std::string &filePath) {
    const size_t slash = filePath.find_last_of('/');
    if (slash == std::string::npos)
        return true;
    std::string dir = filePath.substr(0, slash);
    if (dir.empty())
        return true;

    std::string current;
    if (dir[0] == '/')
        current = "/";

    std::stringstream ss(dir);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty())
            continue;
        if (!current.empty() && current[current.size() - 1] != '/')
            current += "/";
        current += part;
        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
            return false;
    }
    return true;
}

static std::string hexEncodeBytes(const char *text) {
    if (text == nullptr)
        return "";
    static const char *digits = "0123456789abcdef";
    std::string out;
    const unsigned char *p = reinterpret_cast<const unsigned char *>(text);
    while (*p != '\0') {
        out.push_back(digits[(*p >> 4) & 0x0f]);
        out.push_back(digits[*p & 0x0f]);
        p++;
    }
    return out;
}

static std::string escapeTsv(const char *text) {
    if (text == nullptr)
        return "";
    std::string out;
    for (const char *p = text; *p != '\0'; p++) {
        switch (*p) {
        case '\t':
            out += "\\t";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\\':
            out += "\\\\";
            break;
        default:
            out.push_back(*p);
            break;
        }
    }
    return out;
}

static void openCleanOutputDumps(const char *prefix, std::ofstream *tokenDump, std::ofstream *textDump) {
    const std::string base(prefix);
    const std::string tokenPath = base + ".tokens.tsv";
    const std::string textPath = base + ".text";

    if (fileExists(tokenPath))
        throw std::runtime_error("Clean output dump would overwrite existing file: " + tokenPath);
    if (fileExists(textPath))
        throw std::runtime_error("Clean output dump would overwrite existing file: " + textPath);
    if (!ensureParentDirectoryForFileLocal(tokenPath))
        throw std::runtime_error("Cannot create parent directory for clean output dump: " + tokenPath);
    if (!ensureParentDirectoryForFileLocal(textPath))
        throw std::runtime_error("Cannot create parent directory for clean output dump: " + textPath);

    tokenDump->open(tokenPath.c_str(), std::ios::out | std::ios::binary);
    if (!tokenDump->is_open())
        throw std::runtime_error("Cannot open clean token dump: " + tokenPath);
    textDump->open(textPath.c_str(), std::ios::out | std::ios::binary);
    if (!textDump->is_open())
        throw std::runtime_error("Cannot open clean text dump: " + textPath);

    (*tokenDump) << "step\tposition\ttoken_id\tis_eos\tpiece_hex\tpiece_utf8_escaped\n";
}

static void inferenceContinuousBatching(AppInferenceContext *context) {
    const std::vector<std::string> prompts = loadPromptsFromFile(context->args->promptsFile);
    const NnUint maxActive = context->args->decodeCbMaxActive > 0
        ? std::min(context->args->decodeCbMaxActive, context->args->nBatches)
        : context->args->nBatches;
    if (maxActive < 1)
        throw std::runtime_error("decode-cb-max-active must be >= 1");
    // batchPositions[] 를 실제로 쓰는 건 명시 위치 모드(continuous batching)뿐이므로
    // 상한은 nBatches 가 아니라 동시 활성 요청 수 maxActive 에 걸어야 한다.
    // nBatches 에 걸면 배열을 쓰지도 않는 prefill 청크까지 묶인다.
    if (maxActive > MAX_CONTROL_BATCH_POS)
        throw std::runtime_error("decode-cb-max-active exceeds control packet batch position capacity");

    printf("📚 Continuous batching mode: prompts=%zu maxActive=%u\n", prompts.size(), maxActive);

    std::vector<CbRequestState> reqs;
    reqs.reserve(prompts.size());
    for (size_t i = 0; i < prompts.size(); i++) {
        std::vector<int> inputTokensVec(prompts[i].size() + 3);
        int nInputTokens = 0;
        context->tokenizer->encode((char *)prompts[i].c_str(), inputTokensVec.data(), &nInputTokens, true, true);
        if (nInputTokens < 1)
            throw std::runtime_error("Failed to encode prompt");
        if ((NnUint)nInputTokens > context->header->seqLen)
            throw std::runtime_error("Prompt tokens exceed sequence length in prompts-file");
        if ((NnUint)nInputTokens > context->args->steps)
            throw std::runtime_error("Prompt tokens exceed steps in prompts-file");
        CbRequestState st;
        st.tokens.assign(inputTokensVec.begin(), inputTokensVec.begin() + nInputTokens);
        st.nInputTokens = (NnUint)nInputTokens;
        st.pos = 0;
        NnUint maxPos = std::min(context->header->seqLen, context->args->steps);
        if (context->args->decodeCbMaxNewTokens > 0) {
            const NnUint byNewTokens = (NnUint)nInputTokens + context->args->decodeCbMaxNewTokens;
            maxPos = std::min(maxPos, byNewTokens);
        }
        st.maxPos = maxPos;
        st.token = st.tokens[0];
        st.finished = false;
        st.generatedText = "";
        st.generatedTokens = 0;
        reqs.push_back(std::move(st));
    }

    std::ofstream cbDump;
    if (context->args->cleanOutputPrefix != nullptr) {
        const std::string cbPath = std::string(context->args->cleanOutputPrefix) + ".cb.tsv";
        if (fileExists(cbPath))
            throw std::runtime_error("Continuous batching dump would overwrite existing file: " + cbPath);
        if (!ensureParentDirectoryForFileLocal(cbPath))
            throw std::runtime_error("Cannot create parent directory for continuous batching dump: " + cbPath);
        cbDump.open(cbPath.c_str(), std::ios::out | std::ios::binary);
        if (!cbDump.is_open())
            throw std::runtime_error("Cannot open continuous batching dump: " + cbPath);
        cbDump << "request_index\tprompt_tokens\tgenerated_tokens\tfinished\ttext\n";
    }

    NnUint prefillTokensTotal = 0;
    NnUint totalPredTokens = 0;
    Timer wallClock;

    bool cbRowsDumpedIncrementally = false;
    if (maxActive == 1u) {
        cbRowsDumpedIncrementally = true;
        for (size_t r = 0; r < reqs.size(); r++) {
            CbRequestState &req = reqs[r];
            context->inference->setDecodePhase(false);
            const NnUint nPrefillTokens = req.nInputTokens > 0 ? (req.nInputTokens - 1) : 0;
            const NnUint prefillBatchCap = resolvePrefillChunkBatchSize(context->args, nPrefillTokens);
            while (req.pos + 1 < req.nInputTokens) {
                NnUint remainingTokens = req.nInputTokens - 1 - req.pos;
                NnUint batchSize = remainingTokens < prefillBatchCap ? remainingTokens : prefillBatchCap;
                context->inference->setBatchSize(batchSize);
                context->inference->setPosition(req.pos);
                for (NnUint i = 0; i < batchSize; i++)
                    context->inference->setToken(i, req.tokens[req.pos + i]);
                context->inference->forward();
                req.pos += batchSize;
                req.token = req.tokens[req.pos];
                prefillTokensTotal += batchSize;
            }

            context->inference->setBatchSize(1);
            context->inference->setDecodePhase(true);
            context->tokenizer->resetDecoder();
            while (!req.finished && req.pos < req.maxPos) {
                context->inference->setPosition(req.pos);
                context->inference->setToken(0, req.token);
                context->inference->forward();

                int nextToken = context->inference->sampleToken(context->sampler);
                char *piece = context->tokenizer->decode(nextToken);
                if (piece != nullptr)
                    req.generatedText += piece;
                req.token = nextToken;
                req.pos++;
                req.generatedTokens++;
                totalPredTokens++;
                if (context->tokenizer->isEos(nextToken) || req.pos >= req.maxPos)
                    req.finished = true;
            }
            if (cbDump.is_open()) {
                cbDump << r << '\t'
                       << req.nInputTokens << '\t'
                       << req.generatedTokens << '\t'
                       << (req.finished ? 1 : 0) << '\t'
                       << escapeTsv(req.generatedText.c_str()) << '\n';
                cbDump.flush();
            }
        }
    } else {
        // Prefill each request (sequential), then decode in batched rounds.
        context->inference->setDecodePhase(false);
        for (size_t r = 0; r < reqs.size(); r++) {
            CbRequestState &req = reqs[r];
            const NnUint nPrefillTokens = req.nInputTokens > 0 ? (req.nInputTokens - 1) : 0;
            const NnUint prefillBatchCap = resolvePrefillChunkBatchSize(context->args, nPrefillTokens);
            while (req.pos + 1 < req.nInputTokens) {
                NnUint remainingTokens = req.nInputTokens - 1 - req.pos;
                NnUint batchSize = remainingTokens < prefillBatchCap ? remainingTokens : prefillBatchCap;
                context->inference->setBatchSize(batchSize);
                context->inference->setPosition(req.pos);
                for (NnUint i = 0; i < batchSize; i++)
                    context->inference->setToken(i, req.tokens[req.pos + i]);
                context->inference->forward();
                req.pos += batchSize;
                req.token = req.tokens[req.pos];
                prefillTokensTotal += batchSize;
            }
        }

        context->inference->setDecodePhase(true);
        context->tokenizer->resetDecoder();

    NnUint activeCount = (NnUint)reqs.size();
    NnUint rrCursor = 0;
    while (activeCount > 0) {
        std::vector<NnUint> picked;
        picked.reserve(maxActive);
        if (maxActive == 1u) {
            for (NnUint idx = 0; idx < (NnUint)reqs.size(); idx++) {
                if (!reqs[idx].finished && reqs[idx].pos < reqs[idx].maxPos) {
                    picked.push_back(idx);
                    break;
                }
            }
        } else {
            for (NnUint i = 0; i < (NnUint)reqs.size() && picked.size() < maxActive; i++) {
                NnUint idx = (rrCursor + i) % (NnUint)reqs.size();
                if (!reqs[idx].finished && reqs[idx].pos < reqs[idx].maxPos)
                    picked.push_back(idx);
            }
        }
        if (picked.empty())
            break;
        if (maxActive != 1u)
            rrCursor = (picked.back() + 1u) % (NnUint)reqs.size();

        std::vector<NnUint> positions(picked.size());
        for (NnUint i = 0; i < (NnUint)picked.size(); i++) {
            CbRequestState &req = reqs[picked[i]];
            positions[i] = req.pos;
        }
        context->inference->setBatchPositions(positions.data(), (NnUint)picked.size());
        for (NnUint i = 0; i < (NnUint)picked.size(); i++)
            context->inference->setToken(i, reqs[picked[i]].token);

        context->inference->forward();

        for (NnUint i = 0; i < (NnUint)picked.size(); i++) {
            CbRequestState &req = reqs[picked[i]];
            int nextToken = context->inference->sampleTokenAtBatch(context->sampler, i);
            char *piece = context->tokenizer->decode(nextToken);
            if (piece != nullptr)
                req.generatedText += piece;
            req.token = nextToken;
            req.pos++;
            req.generatedTokens++;
            totalPredTokens++;
            if (context->tokenizer->isEos(nextToken) || req.pos >= req.maxPos) {
                req.finished = true;
                activeCount--;
                context->tokenizer->resetDecoder();
            }
        }
    }
    }

    if (cbDump.is_open() && !cbRowsDumpedIncrementally) {
        for (NnUint i = 0; i < (NnUint)reqs.size(); i++) {
            const CbRequestState &req = reqs[i];
            cbDump << i << '\t'
                   << req.nInputTokens << '\t'
                   << req.generatedTokens << '\t'
                   << (req.finished ? 1 : 0) << '\t'
                   << escapeTsv(req.generatedText.c_str()) << '\n';
        }
    }

    NnUint totalWallUs = wallClock.elapsedMicroseconds();
    printf("\nContinuousBatching\n");
    printf("      prompts: %zu\n", reqs.size());
    printf(" prefillTokens: %u\n", prefillTokensTotal);
    printf("  decodeTokens: %u\n", totalPredTokens);
    printf("      totalMs: %3.2f\n", totalWallUs / 1000.0f);
    if (totalPredTokens > 0 && totalWallUs > 0) {
        const float wallDecodeTps = ((float)totalPredTokens) / (totalWallUs / 1000000.0f);
        printf("wall_decode_tps: %3.2f\n", wallDecodeTps);
    }
}

static void inference(AppInferenceContext *context) {
    if (context->args->promptsFile != nullptr) {
        inferenceContinuousBatching(context);
        return;
    }
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0)
        throw std::runtime_error("Number of steps is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    if (nInputTokens > context->header->seqLen)
        throw std::runtime_error("The number of prompt tokens is greater than the sequence length");
    if (nInputTokens > context->args->steps)
        throw std::runtime_error("The number of prompt tokens is greater than the number of steps");

    NnSize sentBytes = 0;
    NnSize recvBytes = 0;
    NnUint evalTotalTime = 0;
    NnUint predTotalTime = 0;
    NnUint evalExecTime = 0;
    NnUint evalSyncTime = 0;
    // EXP-1: prefill 구간의 sync를 wait(peer 대기)/xfer(실제 전송)로 분리 적산.
    unsigned long long prefillSyncWaitUs = 0;
    unsigned long long prefillSyncXferUs = 0;
    // EXP-1/H2: prefill 구간 op 분류별 적산 (attnCore = O(S^2) 항).
    unsigned long long prefillAttnCoreUs = 0;
    unsigned long long prefillAttnProjUs = 0;
    unsigned long long prefillFfnUs = 0;
    unsigned long long prefillNormUs = 0;
    unsigned long long prefillLmHeadUs = 0;
    unsigned long long prefillOtherOpUs = 0;
    bool prefillOpProfilingOn = false;
    NnUint predExecTime = 0;
    NnUint predSyncTime = 0;
    NnSize kvTxTotalBytes = 0;
    NnSize kvRxTotalBytes = 0;
    NnSize activationTxTotalBytes = 0;
    NnSize activationRxTotalBytes = 0;
    NnUint kvMigrationViolationTotal = 0;
    NnUint prefillChunkCount = 0;
    NnUint prefillChunkTokenTotal = 0;
    NnUint queueWaitPrefillUs = 0;
    NnUint queueWaitDecodeUs = 0;
    NnUint prefillSlotCount = 0;
    Timer wallClock;
    NnUint prefillWallUs = 0;
    NnUint ttftWallUs = 0;
    bool hasFirstPredToken = false;

    int token = inputTokens[pos];
    printf("%s\n", context->args->prompt);
    std::ofstream cleanTokenDump;
    std::ofstream cleanTextDump;
    const bool cleanOutputEnabled = context->args->cleanOutputPrefix != nullptr;
    if (cleanOutputEnabled)
        openCleanOutputDumps(context->args->cleanOutputPrefix, &cleanTokenDump, &cleanTextDump);
    const NnUint nPrefillTokens = nInputTokens > 0 ? (NnUint)(nInputTokens - 1) : 0;
    const NnUint prefillBatchCap = resolvePrefillChunkBatchSize(context->args, nPrefillTokens);
    const bool useWave = context->args->wavePipeline && context->args->ppSize > 1;
    const bool tileAlignedSchedule = context->args->tileAligned;
    const long ppSizeForSched = (long)context->args->ppSize;
    const bool prefillUsesSp = context->args->spSize >= 2 && nPrefillTokens >= context->args->spPrefillThreshold;
    const bool useConcurrentPD = context->args->concurrentPrefillDecode && prefillUsesSp;
    if (prefillBatchCap < context->args->nBatches) {
        printf("🔀 Prefill chunking enabled: chunk=%u (maxBatch=%u, threshold=%u)\n",
            prefillBatchCap,
            context->args->nBatches,
            context->args->prefillChunkThreshold);
    }
    if (useWave)
        printf("🌊 Wave pipelining enabled (pp=%u)\n", context->args->ppSize);
    if (context->args->spSize >= 2) {
        printf("🔷 SP prefill-only: %s (threshold=%u, prefillTokens=%u)\n",
            context->args->prefillSpOnly ? "on" : "off",
            context->args->spPrefillThreshold,
            nPrefillTokens);
        if (!prefillUsesSp)
            printf("ℹ️  Prefill tokens below threshold; SP scheduling optimizations are disabled for this request\n");
    }

    // SpGroupScheduler: tracks which SP group is assigned to Prefill vs Decode
    SpGroupScheduler scheduler(context->args->spSize);
    NnUint prefillSpGroup = UINT32_MAX;
    NnUint decodeSpGroup = UINT32_MAX;
    if (useConcurrentPD) {
        prefillSpGroup = scheduler.assignPrefill();
        printf("⚡ Concurrent P/D: SP group %u → PREFILL\n", prefillSpGroup);
    }

    NnUint waveChunkCount = 0; // wave 모드에서 드레인 대기 중인 chunk 수
    context->inference->setDecodePhase(false);
    // EXP-1: 프리필 진입 전에 wait/xfer 누적을 초기화한다(모델 로딩/핸드셰이크 잔여 제거).
    if (context->network != nullptr)
        context->network->resetSyncTimeBreakdown();
    // EXP-1/H2: root에서도 op 단위 프로파일링을 켠다(--stage-timing 1 일 때만).
    if (context->args->stageTiming) {
        context->executor->setStepProfilingEnabled(true);
        prefillOpProfilingOn = true;
    }

    // 스케줄 총량. 패딩이 켜지면 타일 배수로 올림한다.
    const long realTotal = (long)nInputTokens - 1;
    long schedTotal = realTotal;
    // 올림 단위는 청크 크기가 아니라 **GEMM 타일 행 수**다.
    // matmulForward_repack 은 nGemm = rowCount & ~3u 로 4행 단위 GEMM 을 쓰고
    // 나머지 행만 느린 gemv 로 처리한다. 즉 batchSize % 4 == 0 이면 절벽이 없다.
    // 청크 크기로 올리면 짧은 프롬프트에서 크게 낭비된다(13토큰 -> 32위치).
    const long kGemmTileRows = 4;
    if (tileAlignedSchedule)
        schedTotal = ((realTotal + kGemmTileRows - 1) / kGemmTileRows) * kGemmTileRows;

    for (;;) {
        long remainingTokens = schedTotal - (long)pos;
        if (remainingTokens <= 0)
            break;
        NnUint batchSize = remainingTokens < prefillBatchCap
            ? remainingTokens
            : prefillBatchCap;

        // 타일 패딩 (research/14).
        //
        // 커널의 비용 함수 c(B) 는 GEMM 타일 입자도에 대한 계단이다. Q4_0 4행 repack
        // GEMM 이 나머지 행을 gemv 로 처리하기 때문이고, llama.cpp 계열의 표준 구조다.
        // 실측: c(15)=327 ms > c(16)=148 ms — 토큰이 적은데 2.2배.
        //
        // 파이프라인에서는 이 초과분 e 가 **스테이지 수만큼 누적**된다:
        //     스테이지 k 완료 = (m+k-1)c + k·e
        // 앞 스테이지의 지연 위에 자기 초과분이 더해지고 회복되지 않는다.
        // 실측으로 세 번 확인했다 — 나머지 청크를 마지막/1번/중앙 어디에 둬도
        // 손실이 N·e ~ 1,100~1,400 ms 로 동일했다. **안전한 위치가 없다.**
        //
        // 유일한 해법은 비정렬을 없애는 것이다. S=447 은 3x149 라 16 근처 어떤 B 로도
        // 나누어떨어지지 않으므로, 위치를 패딩해 모든 마이크로배치를 B 배수로 만든다.
        // 비용은 토큰 ≤ B-1 개(여기선 1개, 0.2%)이고 이득은 N·e (약 20%)다.
        //
        // prefill 의 로짓은 쓰이지 않는다(디코드가 마지막 실제 토큰부터 시작한다).
        // 패딩 위치의 KV 는 디코드 첫 스텝이 덮어쓴다.
        context->inference->setBatchSize(batchSize);
        // wave 모드에서만 중간 청크의 로짓 송신을 끈다. 비-wave 경로는 청크마다
        // 블로킹 forward() 로 로짓을 기다리므로 항상 보내야 한다.
        context->inference->setLastPrefillChunk(!useWave || (long)pos + (long)batchSize >= schedTotal);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; i++) {
            // 패딩 위치는 마지막 실제 토큰을 반복해 채운다. 출력은 쓰이지 않는다.
            const long src = ((long)pos + (long)i < realTotal) ? (long)pos + (long)i : realTotal - 1;
            context->inference->setToken(i, inputTokens[src]);
        }

        if (useWave) {
            // 겹침 진단: root 가 자기 스테이지(레이어 0..k)만 하는지, 아니면
            // 여기서 워커를 기다리는지 가른다. 자기 몫만 하면 전체 모델 시간의
            // 1/ppSize 여야 한다.
            const auto wt0 = std::chrono::high_resolution_clock::now();
            context->inference->forwardPrefillNoWait();
            const auto wt1 = std::chrono::high_resolution_clock::now();
            // 시간선 계측: root 도 절대 위치를 남긴다. 인접 스테이지의 send/recv
            // 대응으로 노드 간 시계를 정렬하려면 양쪽 모두 필요하다.
            static auto tWave0 = wt0;
            if (prefillChunkCount == 0) tWave0 = wt0;
            if (prefillChunkCount < 60)
                printf("🌊 [R0] mb=%u batch=%u fs=%.1f fe=%.1f\n", prefillChunkCount, batchSize,
                    std::chrono::duration<double, std::milli>(wt0 - tWave0).count(),
                    std::chrono::duration<double, std::milli>(wt1 - tWave0).count());
            waveChunkCount++;
        } else {
            context->inference->forward();
        }
        prefillChunkCount++;
        prefillChunkTokenTotal += batchSize;

        // EXP-1/H2: 청크 forward 직후에만 op breakdown이 유효하다.
        // wave 모드는 forward가 비동기로 끝나지 않으므로 집계 대상에서 제외한다.
        if (prefillOpProfilingOn && !useWave) {
            NnExecutorOpBreakdown ob;
            if (context->executor->getLastForwardOpBreakdown(&ob)) {
                prefillAttnCoreUs += ob.attnCoreUs;
                prefillAttnProjUs += (ob.attnUs >= ob.attnCoreUs) ? (ob.attnUs - ob.attnCoreUs) : 0u;
                prefillFfnUs += ob.ffnUs;
                prefillNormUs += ob.normUs;
                prefillLmHeadUs += ob.lmHeadUs;
                prefillOtherOpUs += ob.otherUs;
            }
        }

        pos += batchSize;
        // 패딩 구간에서는 인덱스를 넘어설 수 있으므로 실제 범위로 자른다.
        token = inputTokens[(long)pos < realTotal ? pos : (NnUint)realTotal];

        if (!useWave) {
            NnTrafficBreakdown breakdown;
            breakdown.kvMigrationViolations = 0;
            if (context->network != nullptr)
                context->network->getTrafficBreakdown(&breakdown);
            if (context->network != nullptr)
                context->network->getStats(&sentBytes, &recvBytes);
            NnUint evalTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
            NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
            printf("🔷️ Eval%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | KV tx/rx %6zu/%6zu kB | Act tx/rx %6zu/%6zu kB | (%d tokens)\n",
                evalTime / 1000,
                syncTime / 1000,
                sentBytes / 1024,
                recvBytes / 1024,
                breakdown.kvTxBytes / 1024,
                breakdown.kvRxBytes / 1024,
                breakdown.activationTxBytes / 1024,
                breakdown.activationRxBytes / 1024,
                batchSize);
            evalExecTime += evalTime;
            evalSyncTime += syncTime;
            evalTotalTime += evalTime + syncTime;
            if (context->network != nullptr) {
                unsigned long long wUs = 0, xUs = 0;
                context->network->getSyncTimeBreakdown(&wUs, &xUs);
                prefillSyncWaitUs += wUs;
                prefillSyncXferUs += xUs;
                context->network->resetSyncTimeBreakdown();
            }
            kvTxTotalBytes += breakdown.kvTxBytes;
            kvRxTotalBytes += breakdown.kvRxBytes;
            activationTxTotalBytes += breakdown.activationTxBytes;
            activationRxTotalBytes += breakdown.activationRxBytes;
            kvMigrationViolationTotal += breakdown.kvMigrationViolations;
        }
    }

    // Wave 모드: 파이프라인에 쌓인 logits 수거 (마지막 chunk의 logits를 logitsPipe에 보존)
    // 패딩으로 pos 가 실제 토큰 수를 넘어섰으면 되돌린다.
    // 디코드는 실제 마지막 토큰 위치에서 시작해야 하고, 패딩 위치의 KV 는
    // 디코드 첫 스텝이 덮어쓴다.
    if ((long)pos > realTotal) {
        pos = (NnUint)realTotal;
        token = inputTokens[pos];
    }

    if (useWave && waveChunkCount > 0) {
        context->inference->drainPrefillLogits(waveChunkCount);
        NnTrafficBreakdown breakdown;
        breakdown.kvMigrationViolations = 0;
        if (context->network != nullptr)
            context->network->getTrafficBreakdown(&breakdown);
        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);
        NnUint evalTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔷️ Wave Eval%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | KV tx/rx %6zu/%6zu kB | Act tx/rx %6zu/%6zu kB | (%u chunks, %u tokens)\n",
            evalTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            breakdown.kvTxBytes / 1024,
            breakdown.kvRxBytes / 1024,
            breakdown.activationTxBytes / 1024,
            breakdown.activationRxBytes / 1024,
            waveChunkCount,
            prefillChunkTokenTotal);
        evalExecTime += evalTime;
        evalSyncTime += syncTime;
        evalTotalTime += evalTime + syncTime;
        if (context->network != nullptr) {
            unsigned long long wUs = 0, xUs = 0;
            context->network->getSyncTimeBreakdown(&wUs, &xUs);
            prefillSyncWaitUs += wUs;
            prefillSyncXferUs += xUs;
            context->network->resetSyncTimeBreakdown();
        }
        kvTxTotalBytes += breakdown.kvTxBytes;
        kvRxTotalBytes += breakdown.kvRxBytes;
        activationTxTotalBytes += breakdown.activationTxBytes;
        activationRxTotalBytes += breakdown.activationRxBytes;
        kvMigrationViolationTotal += breakdown.kvMigrationViolations;
    }
    // WCEP 0c: prefill 종료 시점의 logits 를 그대로 덤프한다 (research/17 §7.7).
    //
    // 왜 필요한가: WCEP 는 bit-identical 을 요구하는데, 그러려면 **현재 baseline 의
    // 누적 순서까지 포함한 reference** 가 있어야 한다. 정수 내적이 수학적으로 같아도
    // block scale 적용 순서와 FP32 누적 순서가 바뀌면 bitwise 결과가 달라진다.
    //
    // DLLAMA_DUMP_LOGITS=<path> 로 켠다. float32 raw, vocabSize 개.
    // 성능 경로에 영향이 없도록 prefill 이 끝난 뒤 한 번만 쓴다.
    if (const char *lp = std::getenv("DLLAMA_DUMP_LOGITS")) {
        FILE *lf = fopen(lp, "wb");
        if (lf != nullptr) {
            fwrite(context->inference->logitsPipe, sizeof(float),
                   context->header->vocabSize, lf);
            fclose(lf);
            printf("  logitsDump: %s (%u floats)\n", lp, context->header->vocabSize);
        }
    }


    prefillWallUs = wallClock.elapsedMicroseconds();
    NnUint prefillEndWallUs = prefillWallUs;

    // SP group role transition: Prefill complete → Decode
    // KV cache stays in place — no data movement between SP groups needed
    if (useConcurrentPD && prefillSpGroup != UINT32_MAX) {
        scheduler.transitionToDecode(prefillSpGroup);
        decodeSpGroup = prefillSpGroup;
        if (context->args->strictKvAffinity && decodeSpGroup != prefillSpGroup) {
            throw std::runtime_error("KV_MIGRATION_VIOLATION: decode lane differs from prefill lane under strict KV affinity");
        }
        printf("⚡ Concurrent P/D: SP group %u → DECODE (KV cache in-place, no transfer)\n", prefillSpGroup);
    }

    fflush(stdout);

    context->inference->setBatchSize(1);
    context->inference->setDecodePhase(true);
    context->tokenizer->resetDecoder();

    const NnUint maxPos = std::min(context->header->seqLen, context->args->steps);
    NnUint firstDecodeStartUs = 0;
    NnUint firstDecodeDoneUs = 0;
    NnUint firstTokenEmitUs = 0;
    NnUint decodeStepsSincePrefillSlot = 0;
    NnUint decodeStepIndex = 0;
    for (; pos < maxPos; pos++) {
        decodeStepIndex++;
        if (firstDecodeStartUs == 0)
            firstDecodeStartUs = wallClock.elapsedMicroseconds();
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        context->inference->forward();
        if (firstDecodeDoneUs == 0)
            firstDecodeDoneUs = wallClock.elapsedMicroseconds();

        token = context->inference->sampleToken(context->sampler);

        char *piece = context->tokenizer->decode(token);
        if (cleanOutputEnabled) {
            cleanTokenDump << decodeStepIndex << '\t'
                           << pos << '\t'
                           << token << '\t'
                           << (context->tokenizer->isEos(token) ? 1 : 0) << '\t'
                           << hexEncodeBytes(piece) << '\t'
                           << escapeTsv(piece) << '\n';
            if (piece != nullptr)
                cleanTextDump << piece;
        }
        if (!hasFirstPredToken) {
            ttftWallUs = wallClock.elapsedMicroseconds();
            firstTokenEmitUs = ttftWallUs;
            hasFirstPredToken = true;
        }
        if (context->args->decodeLogInterval == 0 && piece != nullptr) {
            printf("%s", piece);
            fflush(stdout);
        }

        NnTrafficBreakdown breakdown;
        breakdown.kvMigrationViolations = 0;
        const bool shouldLogDecodeStep = context->args->decodeLogInterval > 0 &&
            (decodeStepIndex % context->args->decodeLogInterval == 0);
        if (shouldLogDecodeStep && context->network != nullptr) {
            context->network->getTrafficBreakdown(&breakdown);
            context->network->getStats(&sentBytes, &recvBytes);
        }

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        if (shouldLogDecodeStep) {
            printf("🔶 Pred%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | KV tx/rx %6zu/%6zu kB | Act tx/rx %6zu/%6zu kB | %s\n",
                predTime / 1000,
                syncTime / 1000,
                sentBytes / 1024,
                recvBytes / 1024,
                breakdown.kvTxBytes / 1024,
                breakdown.kvRxBytes / 1024,
                breakdown.activationTxBytes / 1024,
                breakdown.activationRxBytes / 1024,
                piece == nullptr ? "~" : piece);
            fflush(stdout);
        }
        predExecTime += predTime;
        predSyncTime += syncTime;
        predTotalTime += predTime + syncTime;
        if (shouldLogDecodeStep) {
            kvTxTotalBytes += breakdown.kvTxBytes;
            kvRxTotalBytes += breakdown.kvRxBytes;
            activationTxTotalBytes += breakdown.activationTxBytes;
            activationRxTotalBytes += breakdown.activationRxBytes;
            kvMigrationViolationTotal += breakdown.kvMigrationViolations;
        }

        if (useConcurrentPD && context->args->prefillInterleave) {
            decodeStepsSincePrefillSlot++;
            if (decodeStepsSincePrefillSlot >= context->args->prefillQuota) {
                decodeStepsSincePrefillSlot = 0;
                prefillSlotCount++;
                if (scheduler.hasIdle())
                    printf("🫧 Prefill slot open: decode quota reached (%u)\n", context->args->prefillQuota);
            }
        }
    }
    if (context->args->decodeLogInterval == 0)
        printf("\n");
    if (cleanOutputEnabled) {
        cleanTokenDump.flush();
        cleanTextDump.flush();
    }

    NnUint nEvalTokens = nInputTokens - 1;
    NnUint nPredTokens = pos - nEvalTokens;
    NnUint totalWallUs = wallClock.elapsedMicroseconds();
    NnUint decodeWallUs = totalWallUs >= prefillWallUs ? (totalWallUs - prefillWallUs) : 0;
    const float bubbleRatioEst = (predExecTime + predSyncTime) > 0
        ? (float)predSyncTime / (float)(predExecTime + predSyncTime)
        : 0.0f;
    float evalTotalTimeMs = evalTotalTime / 1000.0;
    float predTotalTimeMs = predTotalTime / 1000.0;
    printf("\n");
    printf("Evaluation\n");
    printf("   nBatches: %d\n", context->args->nBatches);
    if (prefillChunkCount > 0) {
        printf(" prefillChunks: %u (avg %3.2f tok/chunk)\n",
            prefillChunkCount,
            (float)prefillChunkTokenTotal / (float)prefillChunkCount);
    }
    printf("    nTokens: %d\n", nEvalTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nEvalTokens * 1000) / evalTotalTimeMs,
        evalTotalTimeMs / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("    nTokens: %d\n", nPredTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nPredTokens * 1000) / predTotalTimeMs,
        predTotalTimeMs / ((float) nPredTokens));
    printf("Timing\n");
    printf("  prefillMs: %3.2f\n", prefillWallUs / 1000.0f);
    reportOpProfile();      // QCFuse 1단계: op 별 시간 (DLLAMA_OP_PROFILE=1)
    nnCpuOpsReportAttSkipProbe();
    // EXP-1: H1 판정용. syncWait은 peer 대기(straggler), syncXfer는 실제 바이트 이동.
    printf(" syncWaitMs: %3.2f\n", prefillSyncWaitUs / 1000.0f);
    printf(" syncXferMs: %3.2f\n", prefillSyncXferUs / 1000.0f);
    if (prefillOpProfilingOn) {
        // EXP-1/H2 판정용. attnMs는 O(S^2) 항만, gemmMs는 projection+FFN+lm_head.
        const unsigned long long gemmUs =
            prefillAttnProjUs + prefillFfnUs + prefillLmHeadUs;
        printf("     attnMs: %3.2f\n", prefillAttnCoreUs / 1000.0f);
        printf("     gemmMs: %3.2f\n", gemmUs / 1000.0f);
        printf("   attnProjMs: %3.2f\n", prefillAttnProjUs / 1000.0f);
        printf("        ffnMs: %3.2f\n", prefillFfnUs / 1000.0f);
        printf("       normMs: %3.2f\n", prefillNormUs / 1000.0f);
        printf("     lmHeadMs: %3.2f\n", prefillLmHeadUs / 1000.0f);
        printf("      otherMs: %3.2f\n", prefillOtherOpUs / 1000.0f);
    }
    printf("     ttftMs: %3.2f\n", (hasFirstPredToken ? ttftWallUs : prefillWallUs) / 1000.0f);
    printf("   decodeMs: %3.2f\n", decodeWallUs / 1000.0f);
    printf("    totalMs: %3.2f\n", totalWallUs / 1000.0f);
    if (context->args->wallMetrics && nPredTokens > 0 && decodeWallUs > 0) {
        const float wallTpotMs = decodeWallUs / 1000.0f / (float)nPredTokens;
        const float wallDecodeTps = ((float)nPredTokens) / (decodeWallUs / 1000000.0f);
        printf("WallMetrics\n");
        printf(" wall_tpot_ms: %3.2f\n", wallTpotMs);
        printf("wall_decode_tps: %3.2f\n", wallDecodeTps);
    }
    float recvWaitP50Ms = 0.0f;
    float recvWaitP95Ms = 0.0f;
    if (context->inference->getDecodeRecvWaitStats(&recvWaitP50Ms, &recvWaitP95Ms)) {
        printf("CriticalPath\n");
        printf("root_recv_wait_p50_ms: %3.2f\n", recvWaitP50Ms);
        printf("root_recv_wait_p95_ms: %3.2f\n", recvWaitP95Ms);
    }
    printf("TTFTBreakdown\n");
    printf("     prefillEndMs: %3.2f\n", prefillEndWallUs / 1000.0f);
    printf("firstDecodeStartMs: %3.2f\n", firstDecodeStartUs / 1000.0f);
    printf(" firstDecodeDoneMs: %3.2f\n", firstDecodeDoneUs / 1000.0f);
    printf("   firstTokenEmitMs: %3.2f\n", firstTokenEmitUs / 1000.0f);
    printf("TrafficBreakdown\n");
    printf("      kvTxMB: %3.2f\n", kvTxTotalBytes / (1024.0f * 1024.0f));
    printf("      kvRxMB: %3.2f\n", kvRxTotalBytes / (1024.0f * 1024.0f));
    printf("activationTxMB: %3.2f\n", activationTxTotalBytes / (1024.0f * 1024.0f));
    printf("activationRxMB: %3.2f\n", activationRxTotalBytes / (1024.0f * 1024.0f));
    printf("kvMigrationViolations: %u\n", kvMigrationViolationTotal);
    printf("Scheduling\n");
    printf(" bubbleRatioEstDecode: %.3f\n", bubbleRatioEst);
    printf("queueWaitPrefillMs: %3.2f\n", queueWaitPrefillUs / 1000.0f);
    printf(" queueWaitDecodeMs: %3.2f\n", queueWaitDecodeUs / 1000.0f);
    printf(" prefillSlotCount: %u\n", prefillSlotCount);
}

static NnUint readStdin(const char *guide, char *buffer, NnUint size) {
    std::fflush(stdin);
    std::printf("%s", guide);
    if (std::fgets(buffer, size, stdin) != NULL) {
        NnUint length = std::strlen(buffer);
        if (length > 0 && buffer[length - 1] == '\n') {
            buffer[length - 1] = '\0';
            length--;
        }
        return length;
    }
    return 0;
}

static void perplexity(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    printf("Evaluating %d tokens...\n", nInputTokens);

    float totalLogProb = 0.0f;
    NnUint pos = 0;

    // 배치 perplexity.
    //
    // 기본 경로는 setBatchSize(1) 로 토큰을 하나씩 흘린다. 그래서 prefill 의 배치
    // attention 커널(multiheadAttBatch_F32 / multiheadAttFused_F32)을 **전혀 타지
    // 않는다.** 그 경로를 검증하려면 배치 폭을 1보다 크게 줘야 한다.
    //
    // decodePhase(true) 를 유지하는 이유: prefill 에서는 lm_head 가 마지막 행만
    // 계산하므로(research/06 §4) 토큰별 로짓을 얻을 수 없다.
    //
    //   --ppl-batch 32   기존 배치 커널
    //   --ppl-batch 112  융합 커널(auto 는 batchSize > 32 에서 켠다)
    const NnUint pplBatch = std::max(1u, std::min(context->args->pplBatch, context->args->nBatches));
    context->inference->setDecodePhase(true);

    // 꼬리 평가 모드: 앞부분을 배치 prefill(가지치기 적용)로 처리한 뒤,
    // 마지막 evalTail 개 토큰을 batchSize=1 로 하나씩 평가한다.
    //
    // 왜 필요한가: 가지치기는 행을 압축하므로 **logitsPipe 의 행 i 가 더 이상
    // 토큰 i 가 아니다.** 배치 perplexity 는 행↔토큰 대응을 전제하므로 조용히
    // 무효가 된다(실측에서 keep=0.9 인데 perplexity 가 486배로 나왔다 — 생성은
    // 멀쩡했다). 꼬리 평가는 prefill 로 만든 KV 의 품질을 batchSize=1 경로로
    // 재므로 대응 문제가 없고, TTFT 목적의 가지치기가 실제로 해치는 것
    // (이후 예측 능력)을 직접 잰다.
    const NnUint evalTail = context->args->pplEvalTail;
    if (evalTail > 0u && (NnUint)nInputTokens > evalTail + 1u) {
        const NnUint prefillEnd = (NnUint)nInputTokens - evalTail;
        const NnUint width = std::max(1u, std::min(context->args->nBatches, prefillEnd));
        for (NnUint s0 = 0; s0 < prefillEnd; s0 += width) {
            const NnUint n = std::min(width, prefillEnd - s0);
            context->inference->setBatchSize(n);
            context->inference->setPosition(s0);
            for (NnUint i = 0; i < n; i++)
                context->inference->setToken(i, inputTokens[s0 + i]);
            context->inference->forward();
        }
        context->inference->setBatchSize(1);
        for (NnUint pos2 = prefillEnd; pos2 < (NnUint)nInputTokens - 1u; pos2++) {
            context->inference->setPosition(pos2);
            context->inference->setToken(0, inputTokens[pos2]);
            context->inference->forward();
            float *logits = context->inference->logitsPipe;
            softmax_F32(logits, context->header->vocabSize);
            totalLogProb += std::log(std::max(logits[inputTokens[pos2 + 1]], 1e-30f));
        }
        const float avgT = totalLogProb / (float)(nInputTokens - 1 - (int)prefillEnd);
        printf("\nResults\n");
        printf("   perplexity: %f (lower = better)\n", expf(-avgT));
        printf("   avgLogProb: %f\n", avgT);
        return;
    }

    if (pplBatch > 1u) {
        printf("   (배치 perplexity: width=%u)\n", pplBatch);
        const NnUint nEval = (NnUint)(nInputTokens - 1);
        for (NnUint s = 0; s < nEval; s += pplBatch) {
            const NnUint n = std::min(pplBatch, nEval - s);
            context->inference->setBatchSize(n);
            context->inference->setPosition(s);
            for (NnUint i = 0; i < n; i++)
                context->inference->setToken(i, inputTokens[s + i]);
            context->inference->forward();

            for (NnUint i = 0; i < n; i++) {
                float *logits = &context->inference->logitsPipe[(std::size_t)i * context->header->vocabSize];
                softmax_F32(logits, context->header->vocabSize);
                const float prob = logits[inputTokens[s + i + 1]];
                totalLogProb += std::log(std::max(prob, 1e-30f));
            }
        }
        const float avgLogProbB = totalLogProb / (float)(nInputTokens - 1);
        printf("\nResults\n");
        printf("   perplexity: %f (lower = better)\n", expf(-avgLogProbB));
        printf("   avgLogProb: %f\n", avgLogProbB);
        return;
    }

    context->inference->setBatchSize(1);

    for (pos = 0; pos < nInputTokens - 1; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, inputTokens[pos]);
        context->inference->forward();

        float *logits = context->inference->logitsPipe;
        softmax_F32(logits, context->header->vocabSize);

        int targetToken = inputTokens[pos + 1];
        float prob = logits[targetToken];

        totalLogProb += std::log(std::max(prob, 1e-30f));
        printf("%5d / %d, prob=%f\n", pos + 1, nInputTokens - 1, prob);
    }

    float avgLogProb = totalLogProb / (float)(nInputTokens - 1);
    float perplexity = expf(-avgLogProb);

    printf("\n");
    printf("Results\n");
    printf("   perplexity: %f (lower = better)\n", perplexity);
    printf("   avgLogProb: %f\n", avgLogProb);
    printf("   bitPerToken: %f\n", -avgLogProb / std::log(2.0));
}

static void chat(AppInferenceContext *context) {
    const NnUint seqLen = context->header->seqLen;
    char prompt[2048];

    TokenizerChatStops stops(context->tokenizer);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);

    const NnUint sysPromptLength = readStdin("💻 System prompt (optional): ", prompt, sizeof(prompt));
    std::vector<ChatItem> deltaItems;
    if (sysPromptLength > 0)
        deltaItems.push_back(ChatItem{"system", prompt});

    NnUint pos = 0;
    NnUint userPromptLength;
    int token;
    int nInputTokens;
    do {
        do {
            userPromptLength = readStdin("\n👱 User\n> ", prompt, sizeof(prompt));
        } while (userPromptLength == 0);

        deltaItems.push_back(ChatItem{"user", prompt});

        GeneratedChat inputPrompt = templateGenerator.generate(deltaItems.size(), deltaItems.data(), true);
        std::unique_ptr<int[]> inputTokensPtr(new int[inputPrompt.length + 2]);
        int *inputTokens = inputTokensPtr.get();

        bool isStart = pos == 0;
        context->tokenizer->encode((char*)inputPrompt.content, inputTokens, &nInputTokens, isStart, true);

        NnUint userPromptEndPos = (NnUint)std::min<unsigned int>(seqLen, pos + nInputTokens - 1);
        NnUint userPrefillTokens = userPromptEndPos > pos ? (userPromptEndPos - pos) : 0;
        NnUint prefillBatchCap = resolvePrefillChunkBatchSize(context->args, userPrefillTokens);
        NnUint chatPrefillChunkCount = 0;
        if (prefillBatchCap < context->args->nBatches) {
            printf("🔀 Chat prefill chunking enabled: chunk=%u (maxBatch=%u, threshold=%u)\n",
                prefillBatchCap,
                context->args->nBatches,
                context->args->prefillChunkThreshold);
        }
        for (NnUint i = 0; ;) {
            int remainingTokens = userPromptEndPos - pos;
            if (remainingTokens <= 0)
                break;
            NnUint batchSize = remainingTokens < prefillBatchCap
                ? remainingTokens
                : prefillBatchCap;

            context->inference->setBatchSize(batchSize);
            context->inference->setDecodePhase(false);
            context->inference->setPosition(pos);
            for (NnUint j = 0; j < batchSize; j++)
                context->inference->setToken(j, inputTokens[i + j]);

            context->inference->forward();
            chatPrefillChunkCount++;

            i += batchSize;
            pos += batchSize;
            token = inputTokens[i + 1];
        }
        if (chatPrefillChunkCount > 0)
            printf("🔷️ Chat prefill chunks: %u\n", chatPrefillChunkCount);

        context->inference->setBatchSize(1);
        context->inference->setDecodePhase(true);
        context->tokenizer->resetDecoder();

        printf("\n🤖 Assistant\n");
        if (inputPrompt.publicPrompt != nullptr)
            printf("%s", inputPrompt.publicPrompt);

        while (pos < seqLen) {
            context->inference->setPosition(pos);
            context->inference->setToken(0, token);
            context->inference->forward();

            token = context->inference->sampleToken(context->sampler);

            char *piece = context->tokenizer->decode(token);
            EosDetectorType eosType = eosDetector.append(token, piece);
            if (eosType == NOT_EOS || eosType == EOS) {
                char *delta = eosDetector.getDelta();
                if (delta != nullptr) {
                    printf("%s", delta);
                    fflush(stdout);
                }
                eosDetector.reset();
            }
            pos++;
            if (eosType == EOS) break;
        }

        deltaItems.clear();
    } while (pos < seqLen);

    printf("(end of context)\n");
}

static void printUsage() {
    printf("Usage:\n");
    printf("  ./dllama inference --model <path> --tokenizer <path> --prompt <text> --steps <n> [options]\n");
    printf("  ./dllama chat --model <path> --tokenizer <path> [options]\n");
    printf("  ./dllama perplexity --model <path> --tokenizer <path> --prompt <text> [options]\n");
    printf("  ./dllama worker --port <port> [options]\n");
    printf("\n");
    printf("Common options:\n");
    printf("  --nthreads <n>\n");
    printf("  --buffer-float-type <f32|f16|q40|q80>\n");
    printf("  --prompts-file <path>       Enable multi-request continuous batching input (1 prompt per line)\n");
    printf("  --clean-output-prefix <path_prefix>  Write generated tokens/text to <prefix>.tokens.tsv and <prefix>.text without overwriting\n");
    printf("  --decode-cb-max-active <n>  Max active requests per decode step in continuous batching (default: nBatches)\n");
    printf("  --decode-cb-max-new-tokens <n>  Max generated tokens per request in prompts-file mode (0: use --steps)\n");
    printf("  --workers <host:port> [host:port ...]\n");
    printf("  --collective <auto|star|ring>\n");
    printf("  --pp-size <n>\n");
    printf("  --sp-size <n>               Sequence parallel size (default: 1)\n");
    printf("  --pipeline-float-type <f32|f16|q40|q80>  Stage activation transport dtype (default: same as --buffer-float-type)\n");
    printf("  --auto-pp <0|1>             Auto-use pp=nNodes when PP/SP not specified (default: 1)\n");
    printf("  --net-monitor <0|1>         Enable network performance monitor/report (default: 0)\n");
    printf("  --pp-token-only <0|1>       Decode fast path: send sampled token id (argmax) instead of full logits in PP mode (default: 0)\n");
    printf("  --pp-topk <n>               Decode fast path: send top-k logits (id,value) from last PP stage (default: 0)\n");
    printf("  --pipeline-chunk-bytes <n>  Split stage activation payload into chunks of n bytes (default: 0=disabled)\n");
    printf("  --pipeline-delta <0|1>      Send activation delta from previous step when possible (default: 0; lossy on q40/q80/f16)\n");
    printf("  --pipeline-delta-min-bytes <n>  Min payload bytes to enable delta path (default: 4096)\n");
    printf("  --pp-stage-skip-shadow <0|1>  Shadow-only stage skip scoring (default: 0)\n");
    printf("  --pp-stage-skip <0|1>         Execute stage skip routing (default: 0)\n");
    printf("  --pp-stage-skip-target <rank> Skip target PP rank (default: 4)\n");
    printf("  --pp-stage-skip-theta <f>     Gate+verifier threshold on delta_norm (default: 0.10)\n");
    printf("  --pp-stage-skip-alpha <f>     Deterministic bypass strength in [0,1] after gate pass (default: 1.0)\n");
    printf("  --pp-stage-skip-verifier <delta>  Verifier type (v1 supports only: delta)\n");
    printf("  --pp-stage-skip-max-consecutive <n>  Safety cap for consecutive skip decisions (default: 3)\n");
    printf("  --pp-stage-skip-max-reject-streak <n> Safety cap for reject streak before dampening (default: 8)\n");
    printf("  --pp-stage-skip-log <0|1>     Print per-token skip shadow logs on target stage (default: 0)\n");
    printf("  --pp-stage-skip-log-file <path_prefix>  Append per-token skip TSV logs as <path_prefix>.node<N>.tsv\n");
    printf("  --wall-metrics <0|1>        Print wall-clock decode metrics separately from executor metrics (default: 1)\n");
    printf("  --decode-log-interval <n>   Print decode step log every n steps (0: disable, default: 1)\n");
    printf("  --stage-timing <0|1>        Print per-step stage timing (recv/forward/send) for root/workers (default: 0)\n");
    printf("  --prefill-chunk-size <n>\n");
    printf("  --prefill-chunk-threshold <n>\n");
    printf("  --wave-pipeline <0|1>       Enable wave pipelining for prefill (default: 0)\n");
    printf("  --concurrent-pd <0|1>       Enable concurrent Prefill/Decode via SP groups (default: 0)\n");
    printf("  --prefill-interleave <0|1>  Enable quota-based prefill slotting during decode (default: 1)\n");
    printf("  --prefill-quota <n>         Decode steps per prefill slot (default: 8)\n");
    printf("  --strict-kv-affinity <0|1>  Enforce decode-path KV transfer/fetch=0 (default: 0)\n");
    printf("  --allow-kv-migration <0|1>  Allow KV migration when topology/scheduler needs it (default: 1)\n");
    printf("  --prefill-sp-only <0|1>     SP enabled for prefill only (default: 1)\n");
    printf("  --sp-prefill-only <0|1>     Disable SP KV sync during decode (default: 1)\n");
    printf("  --sp-prefill-threshold <n>  Min prompt prefill tokens to enable SP scheduling (default: 256)\n");
    printf("  --help\n");
}

int main(int argc, char **argv) {
    initQuants();
    initSockets();

    int returnCode = EXIT_SUCCESS;
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        if (args.help) {
            printUsage();
            cleanupSockets();
            return EXIT_SUCCESS;
        }
        if (args.mode == nullptr) {
            printUsage();
            throw std::runtime_error("Mode is required");
        }
        if (std::strcmp(args.mode, "inference") == 0) {
            args.benchmark = true;
            runInferenceApp(&args, &inference);
        } else if (std::strcmp(args.mode, "perplexity") == 0)
            runInferenceApp(&args, &perplexity);
        else if (std::strcmp(args.mode, "chat") == 0)
            runInferenceApp(&args, &chat);
        else if (std::strcmp(args.mode, "worker") == 0)
            runWorkerApp(&args);
        else
            throw std::runtime_error("Unsupported mode");
    } catch (const std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        returnCode = EXIT_FAILURE;
    }

    cleanupSockets();
    return returnCode;
}
