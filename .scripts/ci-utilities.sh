#!/bin/bash -eu

########################################
# Report Stylings
########################################
sr=$(printf "\\033[;31m") # red
sg=$(printf "\\033[;32m") # green
sy=$(printf "\\033[;33m") # yellow
sm=$(printf "\\033[;35m") # magenta
sw=$(printf "\\033[;37m") # white
sb=$(printf "\\033[1m")   # bold
sn=$(printf "\\033[0;0m") # reset

########################################
# Helper function for error reporting
########################################
function error_trap {
  echo "${1:-"unknown error"} exit code: $?"
}

########################################
# Demarcations
########################################
function start_section {
  local message="${1}"
  echo
  echo "${sg}${sb}> ${sw}${message}${sn}"
}

########################################
# Makes sure things are equal or fail.
# It requires bash -e flag to cause exit on return
########################################
function assert_equal {
  local x="${1}"
  local y="${2}"
  if [[ ${x} == "${y}" ]] ; then
    echo "Assert: ${x} == ${y}"
  else
    echo "Assertion failed: ${x} == ${y}"
    echo "File ${0}"        # Give name of file.
    return 1
  fi
}

########################################
# Did you really mean that? Like really though?
########################################
function assert_user_confirmation {
  echo "Are you sure you want to do this?"
  echo " In order to proceed type yes"
  read confirmation
  if [[ $confirmation == "yes" ]] ; then
    echo "confirmed"
  else
    echo "canceling"
    return 1
  fi
}

########################################
# Makes sure its a version
########################################
function assert_version {
  local x="${1}"
  if [[ ${x} =~ ^[0-9]*\.[0-9]*\.[0-9]*$ ]] ; then
    echo "Assert: ${x} == version"
  else
    echo "Assertion failed: ${x} == version"
    echo "File ${0}"
    return 1
  fi
}

########################################
# Make sure that the second arg is a
# descendent of the first arg or GAME OVER
########################################
function assert_merged {
  local possible_ancestor="${1}"
  local target="${2}"
  if git merge-base --is-ancestor "${possible_ancestor}" "${target}" ; then
    echo "Assert: ${possible_ancestor} has been merged into ${target}"
  else
    echo "Assertion failed: ${possible_ancestor} merged to ${target}"
    echo "File ${0}"
    return 1
  fi
}

########################################
# Git sanity verification
########################################
function assert_tree_unchanged {
  start_section "Veriyfing current branch is unchanged"
  if git diff-index --quiet HEAD; then
    echo "No changes pending, we are good to go."
  else
    echo "========================================"
    echo "${sr}TREE NOT PRISTINE${sn}"
    echo "===================="
    git status
    echo "===================="
    git diff-index HEAD
    echo "===================="
    git diff-index -p HEAD
    echo "===================="
    echo "This may be caused by a number of factors.  If this condition occurs please get help."
    echo "========================================"
    return 1
  fi
}

########################################
# Calculate new version
########################################
function bump_version {
  local version=$(echo ${1} | cut -d 'v' -f 2)
  local type=${2}

  IFS='.' read -r -a ver_pieces <<< "$version"
  major="${ver_pieces[0]}"
  minor="${ver_pieces[1]}"
  micro="${ver_pieces[2]}"

  if [[ $type == "MAJOR" ]]; then
    major=$((major+1))
    minor=0
    micro=0
  fi

  if [[ $type == "MINOR" ]]; then
    minor=$((minor+1))
    micro=0
  fi

  if [[ $type == "MICRO" ]]; then
    micro=$((micro+1))
  fi

  echo "${major}.${minor}.${micro}"
}
