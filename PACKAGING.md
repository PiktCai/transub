# Packaging Transub

This guide covers the workflow for publishing Transub as a PyPI package and distributing it through Homebrew.

## Prerequisites

### PyPI Account Setup

1. **Register accounts**
   - PyPI: https://pypi.org/account/register/
   - TestPyPI: https://test.pypi.org/account/register/

2. **Configure API tokens** (recommended over passwords)
   - Generate tokens from account settings
   - Create `~/.pypirc`:
   ```ini
   [pypi]
   username = __token__
   password = pypi-AgE...  # Your PyPI API token

   [testpypi]
   username = __token__
   password = pypi-AgE...  # Your TestPyPI API token
   ```

3. **Check package name availability**
   - Visit https://pypi.org/project/transub/ to ensure the name is not taken
   - Once published, version numbers cannot be reused or deleted

## PyPI Release Checklist

1. **Prepare the environment**
   ```bash
   python -m pip install --upgrade build twine
   ```
2. **Clean previous builds**
   ```bash
   rm -rf dist build transub.egg-info
   ```
3. **Update version number**
   - Edit the `version` field in `pyproject.toml`
   - Follow semantic versioning (MAJOR.MINOR.PATCH)
   - Remember: PyPI does not allow re-uploading the same version

4. **Run tests**
   ```bash
   python -m unittest
   ```

5. **Build the distributions**
   ```bash
   python -m build
   ```
   This generates `dist/transub-<version>.tar.gz` and `dist/transub-<version>-py3-none-any.whl`.

6. **Verify the package contents**
   ```bash
   # Check tarball contents
   tar -tzf dist/transub-<version>.tar.gz | grep -E "\.conf|\.md"
   
   # Check wheel contents
   unzip -l dist/transub-<version>-py3-none-any.whl
   
   # Ensure transub-sample.conf is included and transub.conf is excluded
   ```

7. **Test installation in a clean environment**
   ```bash
   # IMPORTANT: Test outside the project directory to avoid import confusion
   cd /tmp
   python -m venv test_env
   source test_env/bin/activate  # Windows: test_env\Scripts\activate
   pip install /path/to/transub/dist/transub-<version>-py3-none-any.whl
   transub --help
   deactivate
   rm -rf test_env
   ```

8. **Upload to TestPyPI**
   ```bash
   twine upload --repository testpypi dist/*
   ```
   Install from TestPyPI and validate:
   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple transub
   transub --help
   ```

9. **Publish to PyPI**
   ```bash
   twine upload dist/*
   ```

10. **Create GitHub release**
    ```bash
    git tag -a v<version> -m "Release v<version>"
    git push origin v<version>
    ```
    Then create a GitHub release at https://github.com/PiktCai/transub/releases/new with:
    - Release notes documenting changes
    - Attached source tarball (optional, GitHub auto-generates it)

### Configuration & Secrets

- The CLI reads its config from the first of:
  1. `TRANSUB_CONFIG` environment variable (path to `transub.conf`)
  2. `./transub.conf` (if present in the current working directory)
  3. `~/.transub/transub.conf`
- Ship only `transub-sample.conf` in the wheel; end users should run `transub init --config ~/.transub/transub.conf` (or rely on the default) to generate their own copy.
- API keys remain outside the config. Make sure the sample config references environment variable names such as `OPENAI_API_KEY`, `OPENROUTER_API_KEY`. Document that users must `export` these before running the CLI (or set them in their shell profile/launchd).
- When testing packaged installs, clear `transub.conf` from the repository root so you exercise the user-level path.

### Metadata Best Practices

Consider enhancing `pyproject.toml` with:

```toml
[project]
# Add author email for user contact
authors = [{ name = "PiktCai", email = "your.email@example.com" }]

# Pin dependency versions to avoid breaking changes
dependencies = [
    "typer>=0.9,<1.0",
    "rich>=13.7,<14.0",
    "pydantic>=2.7,<3.0",
    # ... other deps with upper bounds
]

# Ensure long description works
readme = "README.md"
```

### Pre-Release Checklist

Before publishing, verify:

- [ ] All tests pass (`python -m unittest`)
- [ ] Version number updated in `pyproject.toml`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] `README.md` reflects current features and installation steps
- [ ] `LICENSE` file is present
- [ ] `transub.conf` is **NOT** in the package
- [ ] `transub-sample.conf` **IS** in the package
- [ ] No API keys, tokens, or secrets in any files
- [ ] No `__pycache__`, `.pyc`, or test artifacts in the package
- [ ] Dependencies install correctly in a fresh environment
- [ ] CLI works after installation: `transub --help`, `transub init`
- [ ] Tested on TestPyPI successfully
- [ ] Git working directory is clean (or changes are intentional)

## Homebrew Distribution

1. **Ensure a GitHub release is available**
   - Push the PyPI release tag.
   - Create a GitHub release that includes the source tarball or refers to the PyPI source distribution URL.

2. **Create a tap repository** (if one does not exist):
   ```bash
   gh repo create <org>/homebrew-transub --public
   git clone https://github.com/<org>/homebrew-transub.git
   cd homebrew-transub
   mkdir -p Formula
   ```

3. **Generate a formula skeleton** using the PyPI source distribution:
   ```bash
   brew create --tap <org>/transub https://files.pythonhosted.org/packages/.../transub-<version>.tar.gz
   ```
   Replace the URL with the actual PyPI sdist link.

4. **Define the formula** (example):
   ```ruby
   class Transub < Formula
     include Language::Python::Virtualenv

     desc "CLI tool to transcribe and translate subtitles from videos"
     homepage "https://github.com/PiktCai/transub"
     url "https://files.pythonhosted.org/packages/.../transub-<version>.tar.gz"
     sha256 "<sha256>"
     license "MIT"

     depends_on "python@3.11"

     resource "typer" do
       url "https://files.pythonhosted.org/packages/.../typer-<version>.tar.gz"
       sha256 "<sha256>"
     end

     # Repeat resource blocks for each PyPI dependency.

     def install
       virtualenv_install_with_resources
     end

     test do
       assert_match "Usage", shell_output("#{bin}/transub --help")
     end
   end
   ```

5. **Update Python resources automatically**
   ```bash
   brew update-python-resources Formula/transub.rb
   ```
   This automatically generates `resource` blocks for all PyPI dependencies.

6. **Audit the formula**
   ```bash
   # Check for style and correctness issues
   brew audit --strict --online Formula/transub.rb
   
   # Auto-fix style issues
   brew style --fix Formula/transub.rb
   ```

7. **Test the formula locally**
   ```bash
   # Install from source
   brew install --build-from-source Formula/transub.rb
   
   # Verify installation
   transub --help
   which transub  # Should be in Homebrew's bin directory
   
   # Run formula's test block
   brew test transub
   
   # Uninstall to test again
   brew uninstall transub
   ```

8. **Publish the tap**
   ```bash
   git add Formula/transub.rb
   git commit -m "Add transub formula v<version>"
   git push origin main
   ```

9. **Verify installation from tap**
   ```bash
   # Users install with:
   brew tap <org>/transub
   brew install transub
   
   # Test in a clean shell
   transub --version
   transub --help
   ```

Replace `<org>` with the GitHub organization or username hosting the tap, fill in the correct URLs, versions, and SHA256 hashes, and repeat resource blocks for each dependency listed in `pyproject.toml`.

### Homebrew Notes

- **Python version**: Choose `python@3.11` or `python@3.12` based on Homebrew's current default. Avoid older versions.
- **Testing on multiple macOS versions**: If possible, test on both Intel and Apple Silicon Macs.
- **Submitting to homebrew-core**: For wider distribution, consider submitting to the official `homebrew-core` once the project is stable and has sufficient user interest. Requirements include:
  - Stable version (1.0.0+)
  - No major issues
  - Notable stars/users
  - Passes all Homebrew CI checks
- **Formula updates**: When releasing new versions, update the `url`, `sha256`, and `version` in the formula, then run `brew update-python-resources` again.

## Additional Best Practices

### Maintaining a CHANGELOG

Create `CHANGELOG.md` in the repository root to track version history:

```markdown
# Changelog

All notable changes to Transub will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-10-26

### Added
- Initial release with transcription and translation pipeline
- Support for multiple Whisper backends (local, mlx, cpp, api)
- LLM-based subtitle translation with retry logic
- SRT and VTT output formats
- Interactive configuration wizard
- Resumable pipeline state

### Fixed
- None

### Changed
- None

[0.1.0]: https://github.com/PiktCai/transub/releases/tag/v0.1.0
```

Update this file before each release and reference it in GitHub release notes.

### GitHub Actions Automation (Optional)

Automate PyPI publishing with GitHub Actions. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # For trusted publishing
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build distributions
        run: python -m build
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

Set up `PYPI_API_TOKEN` in repository secrets at:
https://github.com/PiktCai/transub/settings/secrets/actions

### Security Considerations

**Never include in published packages:**
- API keys or tokens
- Personal configuration files (`transub.conf`)
- `.env` files
- Test videos or large media files
- `__pycache__` directories or `.pyc` files
- `.git` directory (should be excluded by default)
- User-specific paths or credentials

**Double-check before each release:**
```bash
# Inspect tarball contents
tar -tzf dist/transub-*.tar.gz

# Look for sensitive patterns
tar -xzf dist/transub-*.tar.gz -O | grep -i -E '(api.?key|password|token|secret)'

# Verify .gitignore is working
git ls-files | grep -E '(\.pyc|__pycache__|\.env|transub\.conf)'  # Should be empty
```

### Version Number Strategy

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes, major rewrites
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

Examples:
- `0.1.0`: Initial release (not yet stable)
- `0.2.0`: Add new Whisper backend
- `0.2.1`: Fix translation retry bug
- `1.0.0`: First stable release with API guarantees

### Troubleshooting Common Issues

**Problem**: `twine upload` fails with authentication error
- **Solution**: Check `~/.pypirc` syntax, regenerate API tokens if needed

**Problem**: Package installs but `transub` command not found
- **Solution**: Verify `[project.scripts]` in `pyproject.toml` and reinstall with `pip install --force-reinstall`

**Problem**: Import errors after installation
- **Solution**: Test in a completely separate directory from the source code to avoid local imports

**Problem**: Homebrew formula fails to install dependencies
- **Solution**: Run `brew update-python-resources Formula/transub.rb` again after any changes to `pyproject.toml`

**Problem**: TestPyPI installation fails with dependency errors
- **Solution**: Use `--extra-index-url https://pypi.org/simple` to allow TestPyPI to fetch dependencies from the main PyPI

**Problem**: `transub-sample.conf` not included in wheel
- **Solution**: Verify `MANIFEST.in` and rebuild with `rm -rf dist build && python -m build`

### Getting Help

- **PyPI packaging guide**: https://packaging.python.org/
- **Homebrew formula cookbook**: https://docs.brew.sh/Formula-Cookbook
- **setuptools documentation**: https://setuptools.pypa.io/
- **twine documentation**: https://twine.readthedocs.io/
