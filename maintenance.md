# Solar Data Tools Repository Maintenance Guide


## 1. Packaging & Releases

### PyPI Release
The release to PyPI is automated via GitHub Actions. To make a release:
- Create a new tagged release in GitHub following semantic versioning.
- Once the tag is pushed, the `build.yml` workflow will build and upload to PyPI.
- If you encounter issues, check the Actions tab for logs. The most likely problems are:
  - Missing or incorrect PyPI credentials in GitHub secrets ([see Trusted Publishers](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)).
  - Version not bumped or duplicate version.
  - Build errors due to code issues.
  - Size limits on PyPI if large files were added to the repo (check [here](https://pypi.org/help/#file-size-limit)).
    You can remove them from the build by adding them to the `MANIFEST.in` file.

### Conda-Forge Release
The conda-forge package is maintained in a separate feedstock repository. The release gets automatically triggered
once a new PyPI version is release (after a few hours).

To update manually:
- Fork the [`solar-data-tools-feedstock`](https://github.com/conda-forge/solar-data-tools-feedstock/).
- Update the version and source hash in `recipe/meta.yaml`. If you are unsure how to get the hash, you can go to:
  ```sh
  https://pypi.org/project/solar-data-tools/<new-version>/#solar_data_tools-<new_version>-py3-none-any.whl
  ```
- If you are adding/removing dependencies, update them in `meta.yaml` as well. Make sure all dependencies are available on conda-forge.
- Increment the build number if the version hasn't changed but you are making other changes (e.g., bug fix in the recipe,
  updated dependencies, or changes in build configuration). Otherwise, set it to 0.
- Submit a PR to the conda-forge feedstock.
- Wait for CI to pass (or fix any CI errors) and merge.

To debug if a release doesn't show up automatically:
- Check the [conda-forge status page](https://conda-forge.org/status/) for any ongoing conda issues or to see if the package
  is queued or errored (under Version Updates).
- Check the feedstock PRs for any errors.
- Ensure the version and hash in `meta.yaml` match the latest PyPI release.
- Communicate with conda-forge maintainers if needed.


## 2. Updating Requirements & Python Version

- Edit `pyproject.toml` for dependencies and minimum Python version. All dependencies should be listed there.
- Update `python_requires` and `classifiers` in `pyproject.toml`, if needed.
- Any new dependencies should be added to the conda-forge recipe as well (make sure they are available on conda-forge,
  otherwise you may need to submit a PR to add them).
- Test with new Python versions using local environments and CI. If you are adding a new Python version, update the
  GitHub Actions workflows in `.github/workflows/test.yml` to include the new version in the test matrix.
- Ensure compatibility by running all tests.
- If you are bumping the minimal Python requirement, you can search for the Python version string being removed:
  ```shell
  git grep -e '3\.10' -- ':*.rst' ':*.md' ':*.toml' ':*.yaml' ':*.py' ':*.yml'
  ```
  You'll also need to manually bump the minimum version in the conda-forge recipe `python_min` [here](https://github.com/conda-forge/solar-data-tools-feedstock/blob/28218e944067d332223aa0cb90e66ecc235df232/recipe/meta.yaml#L3).


## 3. Documentation Website

- Docs are in the `docs/` folder (Sphinx).
- To build locally:
  ```sh
  cd docs
  make html
  ```
- To update ReadTheDocs:
  - Push changes to `main`, `dev`, or make a tagged release. This should trigger a build on ReadTheDocs automatically.
  - If you want to test a build without pushing to main, you can create a new branch and push to that branch by adding
    it to the list of branches in the ReadTheDocs settings, under Versions -> Add version button on top right
    [here](https://app.readthedocs.org/dashboard/solar-data-tools/version/create/).
  - Check [ReadTheDocs build logs](https://readthedocs.org/projects/solar-data-tools/builds/).
  - For major changes, update `index.rst` and API docs.
- Make sure all errors and warnings are resolved when building locally before pushing to the repo. Common issues:
  - Missing dependencies in the `docs/requirements.txt` file.
  - Syntax errors in `.rst` or `.md` files.
  - Broken links (check URLs and internal references).
  - Outdated API docs (regenerate with `sphinx-apidoc` if needed).


## 4. GitHub Workflows

- Workflows are in `.github/workflows/`.
- Our workflows:
  - `test.yml`: runs on all PRs to main. Runs tests on multiple Python versions.
  - `test-build.yml`: runs on all PRs to main. Tries to build the package to check for errors before a release.
  - `lint.yml`: runs on all PRs, pushes, and manually. Lints code with our pre-commit hooks.
  - `build.yml`: runs on tagged releases. Builds and uploads to PyPI (needs PyPI credentials in secrets).
    - `publish-image.yml`: builds and pushes our main Docker image to Dockerhub (under slacgismo) after `build.yml`
      completes successfully (tagged release) (needs Dockerhub credentials in secrets).
- To update any of these workflows:
  - Edit YAML files for new Python versions, dependencies, or steps.
  - Test changes by opening a PR (you may need to adjust the workflow triggers based on your needs).
- Debugging:
  - Check Actions tab for logs.
  - Common issues: missing dependencies, Python version mismatches, secrets not set.


## 6. General Tips

- Always bump the version for releases.
- Test locally before pushing changes.
- Use semantic versioning for releases.
- Keep dependencies up-to-date and remove unused ones.
- Use branches and PRs for all changes.
- Document major changes in the release notes for each version.
- Keep changes up to date on conda-forge feedstock (PyPI releases are automatically updated).
