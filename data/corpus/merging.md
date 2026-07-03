# Merging

Merging combines the histories of two branches. `git merge <branch>` integrates
the named branch into the branch you currently have checked out.

## Fast-forward vs merge commits

If the current branch has not diverged from the branch being merged, Git
performs a **fast-forward**: it simply moves the branch pointer forward, and no
new commit is created. If both branches have new commits, Git creates a **merge
commit** with two parents that ties the histories together.

Pass `--no-ff` to force a merge commit even when a fast-forward is possible,
which keeps an explicit record that a branch was merged.

## Merge conflicts

A conflict occurs when the same lines were changed differently on both branches.
Git marks the conflicting regions in the affected files with `<<<<<<<`,
`=======`, and `>>>>>>>` markers. Edit the files to resolve them, run `git add`
on each resolved file, and then run `git commit` to finish the merge. You can
abort an in-progress merge with `git merge --abort`.
