# Maintaining PDFidelity Translator

## Provenance and license

- Repository: <https://github.com/andygeek/pdfidelity-translator>
- Upstream: <https://github.com/PDFMathTranslate-next/PDFMathTranslate-next>
- Starting commit: `f8dffcf4c3a33b254391d43514439b975ce8d966` (pdf2zh-next 2.9.0).
- Derivative established: **2026-10-10**.
- Original authors, history, copyright notices, and `LICENSE` are retained. This derivative and its modifications use **AGPL-3.0**.

This is an independent public repository with the upstream Git history. The account already has `andygeek/PDFMathTranslate` in the same GitHub fork network, so a second GitHub fork in that network could not be created under the requested name without repurposing the existing repository. That original fork is preserved. The `upstream` remote provides the usual Git fetch/merge workflow for this copy.

The repository is named `pdfidelity-translator`; the distribution remains `pdf2zh-next`, the import remains `pdf2zh_next`, and the commands remain `pdf2zh`, `pdf2zh2`, and `pdf2zh_next`. Existing upstream PyPI packages and hosted Docker images do not contain our modifications.

## Developing with our engine

Clone both repositories as siblings:

```bash
git clone https://github.com/andygeek/pdfidelity-engine.git
git clone https://github.com/andygeek/pdfidelity-translator.git
cd pdfidelity-translator
uv sync --python 3.12
```

`tool.uv.sources` selects `../pdfidelity-engine` as an editable BabelDOC dependency. Changes to either checkout are available in this development environment. Python 3.10 through 3.13 remain supported by the project metadata; Python 3.12 is used in this setup example.

The sibling engine must be present for `uv sync`. `pip install` and installation without uv's source configuration can select the upstream BabelDOC package instead. For our combined development, use the sibling layout and the command above.

Keep provider/model integration, translation configuration, and the CLI/API adapter here. Parsing, formulas, fonts, and PDF reconstruction changes belong in [pdfidelity-engine](https://github.com/andygeek/pdfidelity-engine).

## Container build configuration

The Dockerfile installs this checkout and the PDFidelity engine at commit `e6770a89ee8e7319ab710a57ea9cdc5881ca4e34`. `PDFIDELITY_ENGINE_REV` is the build argument that selects the engine revision. To adopt a later engine change, update the argument default to its full commit SHA. Release builds should use reviewed, immutable commits rather than a moving branch.

The legacy `pymupdf<1.25.3` cap was incompatible with BabelDOC 0.6.4, which requires `pymupdf>=1.26.7`. Both package constraints now agree, and the BabelDOC lower bound is 0.6.4. The Dockerfile no longer replaces our engine with an unpinned PyPI upgrade or uses a random cache-breaking request.

This initial setup has not installed dependencies, run tests, built a container, or changed PDFidelity's production deployment. The dependency and Docker changes still require runtime validation before deployment. GitHub Actions are disabled initially; inherited release/package workflows remain available for later adaptation.

## Incorporating upstream changes

The local project checkout already has `origin` pointing here and `upstream` pointing to PDFMathTranslate-next/PDFMathTranslate-next. For a fresh clone, configure:

```bash
git remote add upstream https://github.com/PDFMathTranslate-next/PDFMathTranslate-next.git
git remote set-url --push upstream https://example.invalid/upstream-read-only
git config remote.pushDefault origin
```

Bring upstream changes into our history:

```bash
git switch main
git fetch upstream --tags
git merge upstream/main
```

Resolve and review conflicts while retaining our modifications, including the engine source and compatible dependency constraints. Commit the resulting changes and push to `origin`. Record each imported upstream revision and behavioral change here. The invalid push URL prevents accidental pushes to upstream; fetching still uses the original GitHub URL.

## Releases and network use

Keep fork changes identifiable and retain AGPL and copyright notices. For a modified version offered over a network, prominently offer users free access to its corresponding source, including required build/install/run scripts. The source must match the deployed revision; a public repository alone does not identify that revision. See `LICENSE`, especially sections 1, 4, 5, and 13.

## Modification record

### 2026-10-10

- Established the public PDFidelity derivative with upstream history intact.
- Added fork identity, provenance, maintenance instructions, and modification notices.
- Updated package maintainer and repository URLs while preserving original authors, package name, version, CLI commands, and license.
- Configured editable sibling-engine development with uv.
- Aligned BabelDOC and PyMuPDF version constraints with our engine baseline.
- Configured Docker to install a pinned commit of our engine and removed the upstream-only upgrade/cache-busting steps.
