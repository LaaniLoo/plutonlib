bump_release() {
    printf "\n"
    read -p "Bump and push? [y = yes, n = no]: " push

    if [[ "$push" == "y" ]]; then
        cz bump
        git tag
        git push origin main --tags
    fi
}

cd ~/plutonlib
bump_release