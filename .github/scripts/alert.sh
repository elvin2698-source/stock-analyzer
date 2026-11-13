#!/usr/bin/env bash
# 알림 이슈 관리: 문제가 있으면 열려 있는 알림 이슈에 댓글(없으면 새로 만듦), 정상으로 돌아오면 닫음.
# 사용: alert.sh <상태: ok|degraded|failed|stale|deps> <설명>
set -u
LEVEL="$1"; DETAIL="${2:-}"
LABEL="자동 알림"
RUN_URL="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
REPO="repos/${GITHUB_REPOSITORY}"
gh api "$REPO/labels" -f name="$LABEL" -f color=d93f0b -f description="자동 분석 문제 알림" >/dev/null 2>&1 || true
OPEN=$(gh api -X GET "$REPO/issues" -f state=open -f labels="$LABEL" -q '.[0].number' 2>/dev/null || true)
case "$LEVEL" in
  ok)
    if [ -n "$OPEN" ]; then
      gh api "$REPO/issues/$OPEN/comments" -f body="정상으로 돌아왔어요. 자동으로 닫을게요. ($RUN_URL)" >/dev/null
      gh api -X PATCH "$REPO/issues/$OPEN" -f state=closed -f state_reason=completed >/dev/null
      echo "알림 이슈 #$OPEN 닫음"
    fi ;;
  *)
    case "$LEVEL" in
      degraded) TITLE="자동 분석 일부 실패 (대시보드는 계속 갱신 중)";;
      failed)   TITLE="자동 분석 실패 (대시보드 갱신 안 됨)";;
      stale)    TITLE="자동 분석이 하루 넘게 실행되지 않음";;
      deps)     TITLE="라이브러리 새 버전에서 문제 발견 (지금은 고정 버전이라 영향 없음)";;
      *)        TITLE="자동 분석 알림";;
    esac
    BODY="**$TITLE** · $(date -u +'%Y-%m-%d %H:%M UTC')

$DETAIL

실행 기록: $RUN_URL"
    if [ -n "$OPEN" ]; then
      gh api "$REPO/issues/$OPEN/comments" -f body="$BODY" >/dev/null && echo "알림 이슈 #$OPEN 에 댓글"
    else
      gh api "$REPO/issues" -f title="$TITLE" -f body="$BODY" -f "labels[]=$LABEL" -q '"알림 이슈 #\(.number) 만듦"'
    fi ;;
esac
