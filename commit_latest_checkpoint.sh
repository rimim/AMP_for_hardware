#!/bin/bash

function usage() {
  echo >&2 "Usage: $0 --remote=<remote amp dir> --task=<task> [--git=<ssh git repo> --name=<name> --video=<video filename> --interval=<minutes>]"
  echo >&2
  echo >&2 "Example: $0 --remote=user@remote.server.com:AMP_for_hardware --git=user@github.com:myrobot_status --task==myrobot"
  echo >&2
}

function createREADME() {
  cat > README.md <<EOF
Latest checkpoint: $CHECKPOINT for $TASK_NAME

Live Stream here:
https://${GITUSER}.github.io/${GITREPO}

Download here:
https://github.com/${GITUSER}/${GITREPO}/raw/main/$VIDEO_FILE
EOF
}

function createINDEX() {
  cat > index.html <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${TASK_NAME} Video</title>
  <meta http-equiv="refresh" content="600">
</head>
<body>
  <h1>${TASK_NAME} Video - ${CHECKPOINT}</h1>
  <video controls autoplay loop muted>
    <source src="https://${GITSERVER}/${GITUSER}/${GITREPO}/raw/main/${VIDEO_FILE}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</body>
</html>
EOF
}

# name: Optional display name for task in README.md and index.html (default same as <task>)
# video: Optional video file name for task (default same as <task>.mp4)
INTERVAL_MINUTES=0
FETCH_ONLY=0

# Parse the arguments
while [[ "$1" =~ ^-- ]]; do
  case "$1" in
    --remote=*)
      REMOTE="${1#*=}"
      ;;
    --git=*)
      GITURL="${1#*=}"
      ;;
    --task=*)
      TASK="${1#*=}"
      ;;
    --video=*)
      VIDEO_FILE="${1#*=}"
      ;;
    --name=*)
      TASK_NAME="${1#*=}"
      ;;
    --interval=*)
      INTERVAL_MINUTES="${1#*=}"
      ;;
    --fetchonly)
      FETCH_ONLY=1
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
  shift
done

# Check if all required arguments are provided
if [[ -z "$REMOTE" || -z "$TASK" ]]; then
  echo "Error: Missing required arguments."
  usage
  exit 0
fi

if ! [[ "$INTERVAL_MINUTES" =~ ^[0-9]+$ ]]; then
    echo "Error: Interval minutes must be a positive integer."
    exit 1
fi
if (( INTERVAL_MINUTES > 0 && INTERVAL_MINUTES < 5 )); then
    echo "Error: Interval minutes must be greater than 5 minutes"
    exit 1
fi
INTERVAL_SECONDS=$((INTERVAL_MINUTES * 60))

if [[ -z "$VIDEO_FILE" ]]; then
  VIDEO_FILE=${TASK}.mp4
fi

if [[ -z "$TASK_NAME" ]]; then
  TASK_NAME=${TASK}
fi

# Regular expression to validate the format user@server:directory
VALID_FORMAT="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+:[a-zA-Z0-9/_-]+$"

# Validate REMOTE and GIT formats
if [[ ! "$REMOTE" =~ $VALID_FORMAT ]]; then
  echo "Error: --remote is not in the correct format (user@server:directory)."
  exit 1
fi

REMOTE_USER="${REMOTE%%@*}"
REMOTE_SERVER="${REMOTE#*@}"
REMOTE_SERVER="${REMOTE_SERVER%%:*}"
REMOTE_DIR="${REMOTE#*:}"

CURRENT_DIR=$PWD

LOCAL_TASK_REPO=
LOCAL_AMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LOCAL_AMP_DIR"

if [ ! -z "$GITURL" ]; then
  if [[ ! "$GITURL" =~ $VALID_FORMAT ]]; then
    echo "Error: --git is not in the correct format (user@server:directory)."
    exit 1
  fi

  # Ensure that the first part is "git@github.com"
  if [[ "$GITURL" != git@github.com:* ]]; then
    echo "Error: Only github repos are supported because, github pages is required"
    exit 1
  fi

  GITUSER=$(echo "$GITURL" | cut -d':' -f2 | cut -d'/' -f1)
  GITREPO=$(echo "$GITURL" | cut -d'/' -f2)
  GITSERVER=github.com

  if [ ! -d "$GITREPO"/.git ]; then
    git clone $GITURL
    if [ $? -ne 0 ]; then
      echo "Error: No such repository or unable to clone the repository at $GITURL"
      echo "Please create a blank repository at $GITURL"
      exit 1
    fi
  fi
  LOCAL_TASK_REPO="$LOCAL_AMP_DIR"/"$GITREPO"
fi

REMOTE_USER_SERVER=${REMOTE_USER}@${REMOTE_SERVER}
REMOTE_TASK_DIR=$REMOTE_DIR/logs/${TASK}
LOCAL_TASK_DIR="$LOCAL_AMP_DIR"/logs/${TASK}
while true;
do
  echo Fetching checkpoint from ${REMOTE_USER_SERVER}
  ssh ${REMOTE_USER_SERVER} 'find '$REMOTE_TASK_DIR' -type d -print -exec stat --format="%Y %n" {} \; | sort -n | tail -1 | cut -d" " -f2' || exit
  LAST_CHECKPOINT_DIR=$(ssh ${REMOTE_USER_SERVER} 'find '$REMOTE_TASK_DIR' -type d -exec stat --format="%Y %n" {} \; | sort -n | tail -1 | cut -d" " -f2')
  LATEST_CHECKPOINT=$(ssh ${REMOTE_USER_SERVER} "find '$LAST_CHECKPOINT_DIR' -name '*.pt' -type f -exec stat --format='%Y %n' {} \; | sort -n | tail -1 | cut -d' ' -f2")
  # latest checkpoint filename
  CHECKPOINT=$(basename $LATEST_CHECKPOINT)
  LOCAL_CHECKPOINT_DIR=$(basename "$LAST_CHECKPOINT_DIR")
  LOCAL_TASK_CHECKPOINT_DIR="$LOCAL_TASK_DIR/$LOCAL_CHECKPOINT_DIR/"

  if [ ! -f "$LOCAL_TASK_CHECKPOINT_DIR"/"$CHECKPOINT" ]; then
    echo Fetching: "$LOCAL_TASK_CHECKPOINT_DIR"/"$CHECKPOINT"
    rsync -avz $REMOTE_USER_SERVER:"$LATEST_CHECKPOINT" "$LOCAL_TASK_CHECKPOINT_DIR" || exit

    if [ ! -z "$LOCAL_TASK_REPO" ]; then
      cd "$LOCAL_TASK_REPO" || exit
      git reset HEAD .
      git restore .
      git clean -fd
      git checkout main || exit

      cd "$LOCAL_AMP_DIR" || exit
      python legged_gym/scripts/record_policy.py --task=$TASK || exit

      days=0
      hours=0
      minutes=0
      CHECKPOINT_NUMBER="${CHECKPOINT#model_}"
      CHECKPOINT_NUMBER="${CHECKPOINT_NUMBER%.*}"
      if [ "$CHECKPOINT_NUMBER" != "0" ]; then
        mtime1=$(stat -c %Y "$LOCAL_TASK_CHECKPOINT_DIR"/model_0.pt)
        mtime2=$(stat -c %Y "$LOCAL_TASK_CHECKPOINT_DIR"/"$CHECKPOINT")
        diff_seconds=$((mtime2 - mtime1))
        days=$((diff_seconds / 86400))
        hours=$(( (diff_seconds % 86400) / 3600 ))
        minutes=$(( (diff_seconds % 3600) / 60 ))
      fi
      CAPTION="${days}d ${hours}h ${minutes}m"
      echo CAPTION: $CAPTION
      ffmpeg -y -i record.mp4 -vf "drawtext=text='$CAPTION':fontcolor=white:fontsize=56:x=(w-text_w)/2:y=h-(text_h*3):alpha='if(lt(t,1),0,if(lt(t,2),t-1,if(lt(t,5),1,if(lt(t,6),1-(t-5),0))))'" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart $LOCAL_TASK_REPO/$VIDEO_FILE || exit
      mkdir -p "$LOCAL_AMP_DIR"/video_history/$LOCAL_CHECKPOINT_DIR
      rsync -auv $LOCAL_TASK_REPO/$VIDEO_FILE "$LOCAL_AMP_DIR"/video_history/$LOCAL_CHECKPOINT_DIR/video_${CHECKPOINT_NUMBER}.mp4

      cd "$LOCAL_TASK_REPO" || exit
      git checkout main || exit

      # Destructive action:
      #  Create a temp_branch
      #  Add and commit video file
      #  Delete main branch
      #  Rename temp branch to main branch
      #  Prune
      cd "$LOCAL_TASK_REPO"
      git checkout --orphan "temp_branch" || exit
      git add "$VIDEO_FILE" || exit
      createREADME || exit
      git add README.md || exit
      git commit -m "Updated $VIDEO_FILE for $CHECKPOINT" || exit
      git branch -D main || exit
      git branch -m main || exit
      git push --force origin main || exit
      git gc --aggressive --prune=all || exit
      git -c gc.reflogExpire=0 -c gc.reflogExpireUnreachable=0 -c gc.rerereresolved=0 \
      -c gc.rerereunresolved=0 -c gc.pruneExpire=now gc

      if git branch --list | grep -q "gh-pages"; then
        # Update github pages index.html with the latest checkpoint name
        git checkout gh-pages || exit
        sed -i "s|<h1>.*</h1>|<h1>Latest $TASK_NAME Video - $CHECKPOINT</h1>|" index.html || exit
        git add index.html || exit
        git commit -m "Commit $CHECKPOINT" || exit
        git push origin gh-pages || exit
      else
        # Create github pages
        git checkout --orphan gh-pages || exit
        git rm -rf . || exit
        createINDEX
        git add index.html || exit
        git commit -m "Commit $CHECKPOINT" || exit
        git push origin gh-pages || exit
      fi 

      git checkout main || exit

      cd "$LOCAL_AMP_DIR" || exit
      LAST_CHECKPOINT=$CHECKPOINT
    fi
  fi
  # Check if INTERVAL_MINUTES is equal to 0
  if [ "$INTERVAL_MINUTES" -eq 0 ]; then
    echo Success
    exit 0
  fi
  echo Waiting $INTERVAL_MINUTES minutes ...
  sleep "$INTERVAL_SECONDS"
done
