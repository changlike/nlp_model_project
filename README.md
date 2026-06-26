



# 깃허브에 nlp_model_project 리포지토리 생성하고, 프로젝트 최초 커밋함
# 해당 리포지토리에서 브렌치 2개 추가 : team01, team02

# 해당 프로젝트 터미널 (또는 Git 메뉴 또는 Git 뷰)에서 현재 브렌치 확인 : *main
git branch
# 깃허브 리포지토리에서 추가한 브렌치 목록 확인
git branch -a
출력:
*main
remotes/origin/team01
remotes/origin/team02

# team01 브렌치로 변경 (checkout)
git checkout -b team01 origin/team01
또는
git switch -c team01 origin/team01
# 현재 브렌치 확인: *team01
