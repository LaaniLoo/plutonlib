#!/bin/bash

sel_dir() {
    read -p "Commit from master or src? [s = source, m = master]: " dir
    if [[ "$dir" == "s" ]]; then
        cd ~/plutonlib/src/plutonlib
        printf "\n"
        echo "selected source"
    fi

    if [[ "$dir" == "m" ]]; then
        cd ~/plutonlib
        printf "\n"
        echo "selected master"
    fi
}

# Add files from src
git_add() { 
    local add_file='y'
    while [[ "$add_file" != "n" ]]; do
        echo "Available files to commit:"
        ls

        printf "\n"
        read -p "Add file? [y = yes, n = no, a = all]: " add_file
        
        #add file prompt
        if [[ "$add_file" == "y" ]]; then
            read -p "Enter filename to add: " file

            #only add if valid file
            if [[ -f "$file" ]]; then
                git add "$file"
                echo "Added $file"
                printf "\n"
                EXEC_COMMIT='y'

            else 
                echo -e "\033[31m$file is not a valid file\033[0m"
            fi
        
        #adds all remaining files and commits them as chore
        elif [[ "$add_file" == "a" ]]; then
            echo "Committing remaining files as misc"
            git add .
            git commit -m "chore: doc cleanup and misc changes"
            EXEC_COMMIT='n' #doesn't execute cz commit
            break
        
        #invalid input
        elif [[ "$add_file" != "y" && "$add_file" != "n" && "$add_file" != "a" ]]; then
            echo -e "\033[31mInvalid input\033[0m"
        fi
    done
}

run_commit() {
    if [[ "$EXEC_COMMIT" == "y" ]]; then
        read -p "Ready to commit? [y = yes, n = no]:" commit
        if [[ "$commit" == "y" ]]; then
            cz_commit     
        fi
    fi
}
cz_commit() {
    read -p "Finalize commit in vim? [y = yes, n = no]: " use_vim
    if [[ "$use_vim" == "y" ]]; then
        printf "\n"
        echo "Using vim to finalize commit"
        cz commit -e
    else
        printf "\n"
        echo "Using standard commitizen"
        cz commit
    fi
}  

sel_dir
git_add
run_commit
