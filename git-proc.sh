#!/bin/bash

# Bold High Intensity Colors
BIBlack="\033[1;90m"      # Black
BIRed="\033[1;91m"        # Red
BIGreen="\033[1;92m"      # Green
BIYellow="\033[1;93m"     # Yellow
BIBlue="\033[1;94m"       # Blue
BIPurple="\033[1;95m"     # Purple
BICyan="\033[1;96m"       # Cyan
BIWhite="\033[1;97m"      # White

# High Intensity Backgrounds
On_IBlack="\033[0;100m"   # Black
On_IRed="\033[0;101m"     # Red
On_IGreen="\033[0;102m"   # Green
On_IYellow="\033[0;103m"  # Yellow
On_IBlue="\033[0;104m"    # Blue
On_IPurple="\033[0;105m"  # Purple
On_ICyan="\033[0;106m"    # Cyan
On_IWhite="\033[0;107m"   # White

# Standard Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}Done. Checking status:${NC}"
git status
echo

# Display action
echo -e "${RED}Performing git pull...${NC}"
git pull

# Check current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "${YELLOW}Current branch: ${BRANCH}${NC}"

# Handle branch logic
if [ "$BRANCH" != "master" ] && [ "$BRANCH" != "main" ]; then
    echo -e "${RED}You are not on 'master' or 'main'. Please switch to the appropriate branch!${NC}"
    exit 1
elif [[ $(git status --porcelain) ]]; then
    echo -e "${YELLOW}Uncommitted changes detected.${NC}"
    now=$(date)
    echo -e "${GREEN}Staging changes...${NC}"
    git add .
    echo -e "${GREEN}Committing changes...${NC}"
    git commit -m "Updated at $now"
    echo -e "${GREEN}Pushing changes...${NC}"
    git push
else
    echo -e "${GREEN}No changes to commit. Your branch is up to date.${NC}"
fi

# Final status
echo -e "${CYAN}Done. Checking status:${NC}"
git status
echo

