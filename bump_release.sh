bump_release() {
    printf "\n"
    read -p "Bump and push? [y = yes, n = no]: " push

    if [[ "$push" == "y" ]]; then
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