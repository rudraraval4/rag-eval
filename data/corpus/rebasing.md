# Rebasing

Rebasing moves a sequence of commits to a new base commit, producing a linear
history. `git rebase <base>` replays the commits of the current branch, one by
one, on top of `<base>`.

## Rebase vs merge

Merging preserves the exact history and records where branches came together
with a merge commit. Rebasing rewrites history to make it linear, which can be
easier to read but changes commit hashes. Because it rewrites commits, you
should not rebase commits that have already been pushed and shared with others.

## Interactive rebase

`git rebase -i <base>` opens an editor where you can reorder, edit, squash, or
drop commits. Squashing combines several commits into one, which is useful for
cleaning up a messy branch before merging.

## The golden rule

Never rebase commits that exist outside your local repository and that other
people may have based work on. Rewriting shared history forces everyone else to
reconcile diverged histories.
