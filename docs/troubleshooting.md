# 커밋 아이덴티티 오염

> 배경: bench/hf_bench.py 파일을 작성하는 도중 git에서 설정한 ID 문제로 커밋메세지에 다른 사람의 계정으로 계속 commit이 되고 있는 문제를 발견

local-llm-lab 레포의 커밋이 `gregnewman`이라는 모르는 계정으로 표시되던 문제.

1. Author는 Local에서 Committer는 Global에서 가져오는 문제

- 이때, Local과 Global의 설정 모두 달라서 문제가 더 복잡해짐

2. github 에서는 커밋에 적혀있는 이름을 무시하고 이메일로 계정을 조회하는데,

- <GH-ID>+YB-nt@users.noreply.github.com 해당 값을 아무값을 넣어도 무관하다고 생각하여
- 이와 같은 문제가 발생하여 commit 메세지가 다른계정으로 진행되었다.

위의 문제를 해결하기 위해서 실제로 동작하였던 명령어

## 1. 진단 — 히스토리에 어떤 아이덴티티가 있는지 확인

```bash
git log --format='%an <%ae> | %cn <%ce>' | sort -u
```

이 명령이 전체 작업의 축이었다. Author와 Committer를 함께 보여줘서 불일치를 바로 드러내고, 재작성 후 검증에도 그대로 썼다. 한 줄만 나오면 정리된 것.

원격 확인은 ref를 붙여서:

```bash
git log origin/main --format='%an <%ae> | %cn <%ce>' | sort -u
```

## 2. 실제 GitHub 사용자 ID 확인 — 문제 해결의 열쇠

```bash
curl -s https://api.github.com/users/YB-nt | grep '"id"'
```

74981759를 얻은 시점이 전환점이었다. 이전까지의 시도가 모두 실패한 이유가 여기서 드러났다.

## 3. config 교정

```bash
git config --global user.name "YB-nt"
git config --global user.email "74981759+YB-nt@users.noreply.github.com"

git config --local user.name "YB-nt"
git config --local user.email "74981759+YB-nt@users.noreply.github.com"
```

Global과 Local을 같은 값으로 통일. 이걸 히스토리 재작성보다 먼저 해야 다음 커밋이 다시 오염되지 않는다.

## 4. 히스토리 재작성 — git filter-repo

```bash
cat > /tmp/mailmap.txt << 'EOF'
YB-nt <74981759+YB-nt@users.noreply.github.com> YB-nt <2026+YB-nt@users.noreply.github.com>
YB-nt <74981759+YB-nt@users.noreply.github.com> ybnt <ybnt.dev@gmail.com>
EOF

git filter-repo --mailmap /tmp/mailmap.txt --force
```

mailmap 형식은 새이름 <새이메일> 기존이름 <기존이메일> — 왼쪽이 목적지
이 방향을 뒤집어서 한 번 실패했다.

--force는 이미 재작성된 레포에서 다시 돌릴 때 필요하다.

## 5. remote 복구 후 push

```bash
git remote add origin https://github.com/YB-nt/local-llm-lab.git
git push --force origin main
```

filter-repo는 실행 시 remote를 의도적으로 제거(잘못된 히스토리를 자동으로 밀어버리지 않게 하려는 안전장치)
매번 다시 붙여야 한다.

---

시도 하였지만 효과 X

- git filter-branch : 워킹트리가 꺠끗해야지 실행, git 공식적으로 권장 X
- 이름만 변경 : Github가 이메일로 계정으로 조회를 하기 떄문에 무의미
- mailmap 방향 반대로 작성 : 오히려 모든 커밋이 `gregnewman` 로 변경

---

## 사후처리

PR #2는 히스토리 재작성으로 base가 어긋나 충돌 상태가 됐다.
커밋 내용은 이미 main에 있으므로 merge하지 않고 Close.

대응 이후 17시간 이후에 Support에 문의를 진행

아래와 같은 내용으로 답변을 받았다.

Contributors 목록은 Git 히스토리가 아니라 GitHub 서버 캐시인데,
재작성 후 갱신까지 최대 24시간 걸린다. 그 이후에도 남으면 Support 재문의

---

## 알게된 것

- GitHub는 이메일로 계정을 판단한다. 이름은 표시에 쓰이지 않는다.
- noreply 주소의 숫자는 계정 고유 ID다. Settings → Emails에서 정확한 값을 확인해야 한다.
- 추측하거나 남의 것을 복사하면 커밋이 엉뚱한 계정에 귀속된다.
- 남의 ID가 config에 들어 있었다는 건 dotfiles를 통째로 가져다 썼다는 정황이다.
- ~/.gitconfig의 signingkey, credential.helper, url.*.insteadOf 항목도 점검
- 히스토리 재작성 전에 백업. 디렉터리 통째 복사가 가장 확실
- filter-repo가 .git/filter-repo/에 ref-map을 남기긴 하지만 복구가 번거롭다.
