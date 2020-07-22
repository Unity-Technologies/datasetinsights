Building documentation
======================

Run the following commands from `docs` directory.

Automatic generate of Sphinx sources using [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)
```bash
make apidoc
```

Build html files

```bash
make html
```

You can browse the documentation by opening `build/html/index.html` file directly in any web browser.

Cleanup build files
```bash
make clean
```


Known issues
------------

When using commonmark to convert markdown files to rst files, links to headers cannot have `_` or `.`.
If the header has either of those characters, they should be replaced by dashes `-`.
e.g. if you have a header `#### annotation_definitions.json` in the markdown file, to link to that header the markdown needs to be `[click link](#annotation-definitions-json)`
