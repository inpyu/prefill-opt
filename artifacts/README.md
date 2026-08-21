# artifacts/

측정 산출물의 불변 보관소. 논문의 모든 수치는 여기의 run-id 를 인용해야 한다.

    bash prefill_bench/snapshot.sh j2_baxis pl_*.tsv /path/to/j2_all.tsv

각 `<run-id>/` 는 `raw/`(원자료), `derived/`, `manifest.json`(git SHA·바이너리 md5),
`dirty.patch`(커밋 안 된 변경), `hosts.tsv`(코어·governor·주파수·온도),
`environment.txt`, `checksums.txt` 를 담는다.

**규칙**: 한 번 쓴 run-id 는 수정하지 않는다. 재측정은 새 run-id 로 만든다.
