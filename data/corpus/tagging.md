# Tagging

Tags mark specific points in history as important, most commonly to record
release versions such as `v1.0.0`.

## Lightweight vs annotated tags

A **lightweight tag** is just a name pointing at a commit. An **annotated tag**
is a full object stored in the repository with a tagger name, email, date, and
message, and it can be signed with GPG. Annotated tags are recommended for
releases because they carry metadata; create one with
`git tag -a v1.0.0 -m "Release 1.0.0"`.

## Listing and inspecting tags

`git tag` lists tags. `git show <tag>` shows the tag's metadata (for annotated
tags) and the commit it points to.

## Pushing tags

Tags are not pushed by `git push` automatically. Push a single tag with
`git push origin <tag>`, or push all tags at once with `git push --tags`.
