command -v git-cliff >/dev/null 2>&1 || {
    echo "❌ git-cliff not found! Please install it before bumping a release."
    exit 1
}

bump_release() {
    printf "\n"
    read -p "Bump and push? [y = yes, n = no]: " push

    if [[ "$push" == "y" ]]; then

        next_version=$(cz bump --get-next)
        echo "Next version will be: v$next_version"
        # git-cliff --tag $next_version -o CHANGELOG.md
        # git-cliff --unreleased --tag $next_version -o CHANGELOG.md --append
        git-cliff --unreleased --tag $next_version --prepend CHANGELOG.md

        git add CHANGELOG.md
        git commit -m "docs: update changelog"


        cz bump
        git tag
    fi

    read -p "Ready to push? (verify tag) [y = yes, n = no]: " ready
    if [[ "$ready" == "y" ]]; then
        git push origin main --tags
    fi

}

cd ~/plutonlib
bump_release