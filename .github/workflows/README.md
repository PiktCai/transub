# GitHub Actions Workflows

## PyPI Publishing Workflow

The `publish.yml` workflow automatically publishes packages to PyPI when a new GitHub release is created.

### Setup Instructions

#### Option 1: Trusted Publishing (Recommended)

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/publishing/)
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI project name**: `transub`
   - **Owner**: `PiktCai` (or your GitHub username/org)
   - **Repository name**: `transub`
   - **Workflow filename**: `publish.yml`
4. Click "Add"
5. The workflow will automatically use trusted publishing - no secrets needed!

#### Option 2: API Token (Alternative)

If you prefer using an API token:

1. Generate an API token at [PyPI Account Settings](https://pypi.org/manage/account/token/)
2. Go to GitHub repository → Settings → Secrets and variables → Actions
3. Add a new secret named `PYPI_API_TOKEN` with your token value
4. Update `publish.yml` to use:
   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

### How It Works

1. When you create a GitHub release (tagged with `v*.*.*`), the workflow triggers
2. It runs all tests to ensure code quality
3. Builds the distribution packages (wheel and sdist)
4. Verifies package contents (excludes test files, includes sample config)
5. Publishes to PyPI automatically

### Manual Testing

To test the workflow without creating a release:

```bash
# Create a test release (will trigger workflow)
git tag -a v0.0.0-test -m "Test release"
git push origin v0.0.0-test
```

Then delete the test tag:
```bash
git tag -d v0.0.0-test
git push origin :refs/tags/v0.0.0-test
```

