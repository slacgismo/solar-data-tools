#!/bin/bash -eu
set -o errtrace

########################################
# Cache the root dir and the latest git
# tag and commit hash
########################################
BUILD_ROOT=$(git rev-parse --show-toplevel)
LAST_RELEASE_TAG="$(git tag --list "v*[0-9]" --sort=version:refname | tail -1 )"
LAST_RELEASE_COMMIT="$(git rev-list -n 1 "${LAST_RELEASE_TAG}")"

# import release utilities
source "$BUILD_ROOT"/.scripts/ci-utilities.sh

# facility for error reporting
trap 'error_trap "$0 line $LINENO"' ERR

########################################
# Increment the version, cut the tag and push it
########################################
function push_release_tag  {
    local version
    version=$LAST_RELEASE_TAG
    new_version=$(bump_version $version $BUMP_TYPE)

    start_section "Tagging master with v${new_version}"
    assert_user_confirmation
    git tag -a "v${new_version}" -m "perform-release on v${new_version}"
    git push origin "v${new_version}"
}

########################################
# Release a new version of the code by
# tagging the latest commit in master with
# bump in major, minor or micro version
########################################
function main {
    start_section "Validating SEMVER Cut"
    if [ "$#" == 0 ]; then
        echo "No version specified. Please specify the version bump with either major, minor or micro."
        exit
    fi

    BUMP_TYPE=$(printf '%s\n' "$1" | awk '{ print toupper($0) }')

    if [ "$BUMP_TYPE" == "MAJOR" ]; then
        echo "Releasing a new MAJOR version"
    elif [ "$BUMP_TYPE" == "MINOR" ]; then
        echo "Releasing a new MINOR version"
    elif [ "$BUMP_TYPE" == "MICRO" ]; then
        echo "Releasing a new MICRO version"
    else
        echo "Invalid bump type. Please use either major, micro or minor"
        exit
    fi

    export BUMP_TYPE

    start_section "Latest information"
    echo "Last Release Tag: ${LAST_RELEASE_TAG}"
    echo "Last Release Commit: ${LAST_RELEASE_COMMIT}"

    start_section "Validating that you are on master"
    pushd "$BUILD_ROOT"
    assert_equal $(git rev-parse --abbrev-ref HEAD) master
    git pull
    assert_tree_unchanged
    assert_merged "${LAST_RELEASE_COMMIT}" origin/master
    popd

    push_release_tag
}

main "$@"
