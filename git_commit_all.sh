#!/usr/bin/env bash
# git_commit_all.sh
# 把 DiracNet_V3 仓库中"重要内容"（不含 logs / checkpoints / data_cache / 大型归档）暂存并提交。
#
# 行为：
#   1. 在仓库根（默认 /home/chaos/workspace2/DiracNet_V3）执行。
#   2. git add -A 后，额外显式 unstage 几个大文件（即使 .gitignore 已覆盖也保险）：
#        - rc_pinn_art_project/stage_a_500ep_results.tar.gz
#   3. 显示 staged 内容，等待用户确认；输入 y/Y 才真正 commit。
#   4. 不 push；如需推送，手动 `git push`。
#
# 用法：
#   bash git_commit_all.sh                  # 默认 commit message
#   bash git_commit_all.sh "自定义 message"   # 自定义 commit message
#   bash git_commit_all.sh --dry-run        # 只显示会被 add 的文件，不 commit

set -euo pipefail

# ---------- 参数 ----------
DRY_RUN=0
MSG=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) MSG="$arg" ;;
  esac
done

# ---------- 工作区 ----------
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
echo "==> 工作区: $REPO_ROOT"
echo "==> 分支:   $(git rev-parse --abbrev-ref HEAD)"

# ---------- 1) 健康检查 ----------
if [[ -n "$(git status --porcelain)" ]]; then :; fi
if ! git diff --cached --quiet 2>/dev/null; then
  echo "==> 注意：已有 staged 内容（来自之前 add），将一并提交。"
fi

# ---------- 2) git add -A ----------
echo "==> git add -A ..."
git add -A

# ---------- 3) 显式排除（防御性） ----------
# 这些路径即使在 .gitignore 中已覆盖，再 reset 一次更稳。
EXCLUDE_PATHS=(
  "rc_pinn_art_project/stage_a_500ep_results.tar.gz"
)
for p in "${EXCLUDE_PATHS[@]}"; do
  if git ls-files --error-unmatch --stage -- "$p" >/dev/null 2>&1; then
    git rm --cached -- "$p" >/dev/null
    echo "    排除 (已 staged): $p"
  fi
done

# 防御性二次确认：检查大文件 / 已忽略目录中是否有意外被 add
echo "==> 自检：staged 中是否含 logs/ checkpoints/ data_cache/ 大型归档？"
if git diff --cached --name-only | grep -E '^(.*/)?(logs|checkpoints|data_cache)/' >/dev/null; then
  echo "    警告：发现 logs/checkpoints/data_cache 路径混入 staged！"
  git diff --cached --name-only | grep -E '^(.*/)?(logs|checkpoints|data_cache)/' | sed 's/^/      /'
  echo "    已中止，请检查 .gitignore 配置。"
  exit 1
fi
if git diff --cached --name-only | grep -E '\.(msgpack|pt|tar\.gz)$' >/dev/null; then
  echo "    警告：发现大文件后缀（.msgpack/.pt/.tar.gz）混入 staged："
  git diff --cached --name-only | grep -E '\.(msgpack|pt|tar\.gz)$' | sed 's/^/      /'
  echo "    已中止。"
  exit 1
fi
echo "    自检通过：staged 内容不含 logs/checkpoints/data_cache 及大型归档。"

# ---------- 4) dry-run 出口 ----------
if [[ $DRY_RUN -eq 1 ]]; then
  echo "==> --dry-run 模式：以下文件将被 commit："
  git diff --cached --name-only
  echo "==> （未 commit）"
  exit 0
fi

# ---------- 5) 显示 staged 概览 ----------
STAGED_COUNT=$(git diff --cached --name-only | wc -l | tr -d ' ')
echo "==> 共 $STAGED_COUNT 个文件 staged。"
echo "==> 前 30 个 staged 文件："
git diff --cached --name-only | head -30 | sed 's/^/    /'
if [[ $STAGED_COUNT -gt 30 ]]; then
  echo "    ...（其余 $((STAGED_COUNT - 30)) 个省略）"
fi

# ---------- 6) 确认 ----------
echo
if [[ -z "$MSG" ]]; then
  DEFAULT_MSG="chore: snapshot working state $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "==> 将使用默认 commit message："
  echo "    $DEFAULT_MSG"
  MSG="$DEFAULT_MSG"
fi

read -r -p "==> 确认 commit？[y/N] " ans
case "$ans" in
  y|Y) ;;
  *) echo "==> 取消，未 commit。"; exit 0 ;;
esac

# ---------- 7) commit ----------
git commit -m "$MSG"
echo "==> commit 完成。HEAD: $(git rev-parse --short HEAD)"
echo "==> 如需推送：git push origin $(git rev-parse --abbrev-ref HEAD)"
