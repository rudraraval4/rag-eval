# Stashing

`git stash` sets aside uncommitted changes so you can switch context with a clean
working tree, then reapply them later. It is useful when you need to switch
branches but are not ready to commit.

## Saving and restoring

`git stash push` (or just `git stash`) saves both staged and unstaged changes to
tracked files and reverts the working tree to the last commit. `git stash pop`
reapplies the most recent stash and removes it from the stash list. `git stash
apply` reapplies it but keeps it in the list.

## Managing the stash list

Stashes are stored on a stack. `git stash list` shows all saved stashes, each
identified like `stash@{0}`. `git stash drop stash@{0}` deletes one stash, and
`git stash clear` removes all of them.

## Including untracked files

By default `git stash` ignores untracked files. Use `git stash -u` to include
untracked files, or `git stash -a` to include ignored files as well.
