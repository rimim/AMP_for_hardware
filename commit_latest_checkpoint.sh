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
  <title>${TASK_NAME}</title>
  <meta http-equiv="refresh" content="600">
  <style>
    body {
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      height: 100vh;
      box-sizing: border-box;
    }
    video {
      width: 100%;
      height: auto;
      max-height: 50vh;
      margin-bottom: 10px;
      outline: none;
    }
    img {
      width: 100%;
      height: auto;
      cursor: pointer;
      margin-bottom: 10px;
    }
    a {
      display: block;
      width: 100%;
    }
    h1, h2 {
      margin: 0;
      text-align: center;
    }
  </style>
</head>
<body>
  <h1>BDX Duckling</h1>
  <video autoplay loop muted playsinline>
    <source src="https://${RAWGITSERVER}/${GITUSER}/${GITREPO}/raw/main/${VIDEO_FILE}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <h2>Checkpoint ${CHECKPOINT_NUMBER}</h2>
  <a href="https://${RAWGITSERVER}/${GITUSER}/${GITREPO}/raw/main/plot.png" target="_blank">
    <img src="https://${RAWGITSERVER}/${GITUSER}/${GITREPO}/raw/main/plot.png" alt="${TASK_NAME} Plot">
  </a>
  <a href="https://${RAWGITSERVER}/${GITUSER}/${GITREPO}/main/tensorboard.png" target="_blank">
    <img src="https://${RAWGITSERVER}/${GITUSER}/${GITREPO}/rimim/duck_gym/main/tensorboard.png" alt="Tensorboard">
  </a>
  <video autoplay loop muted playsinline>
    <source src="https://${RAWGITSERVER}/${GITUSER}/${GITREPO}/raw/main/${SIM2SIM_VIDEO_FILE}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <script>
    const video = document.querySelector('video');
    video.addEventListener('click', function() {
      if (!document.fullscreenElement) {
        if (video.requestFullscreen) {
          video.requestFullscreen();
        } else if (video.webkitRequestFullscreen) {
          video.webkitRequestFullscreen();
        } else if (video.msRequestFullscreen) {
          video.msRequestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
          document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
          document.msExitFullscreen();
        }
      }
    });
</body>
</html>
EOF
}

function fetch_tensordata() {
  local format=$1
  local report=$2
  local output=$3
  if [[ -z "$HISTORY_DIR" ]]; then
    echo "HISTORY_DIR must be defined"
    return
  fi
  mkdir -p "$HISTORY_DIR"/reports
  wget -q "http://localhost:6006/data/plugin/scalars/scalars?run=${TASK}/${LOCAL_CHECKPOINT_DIR}&tag=${report}&format=${format}" -O "$HISTORY_DIR"/reports/${output}.${format}
  if [ ! "$HISTORY_DIR"/reports/${output}.${format} ]; then
    echo "The report does not exist or is empty. Removing the file."
    rm -f "$HISTORY_DIR"/reports/${output}.${format}
  fi
}

function monitor_new_model_files() {
  local directory_to_watch="$1"
  local file_pattern="model_*.pt"
  local last_checked_file=""

  echo "Monitoring directory: $directory_to_watch for new files matching pattern: $file_pattern"
  if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -m -e create --format '%f' "$directory_to_watch" | while read -r new_file; do
      if [[ $new_file == $file_pattern ]]; then
        echo "New file detected: $new_file"
        # Perform your desired action here, e.g., processing the new file
        echo "Processing $new_file..."
        sleep 2
        pkill -P $$ inotifywait
        break
      fi
    done
  else
    echo Waiting $INTERVAL_MINUTES minutes ...
    sleep $INTERVAL_MINUTES
  fi
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
if [[ -z "$TASK" ]]; then
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
SIM2SIM_VIDEO_FILE=sim2sim-${VIDEO_FILE}

if [[ -z "$TASK_NAME" ]]; then
  TASK_NAME=${TASK}
fi

# Regular expression to validate the format user@server:directory
VALID_FORMAT="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+:[a-zA-Z0-9/_-]+$"

if [[ ! -z "$REMOTE" ]]; then
  # Validate REMOTE and GIT formats
  if [[ ! "$REMOTE" =~ $VALID_FORMAT ]]; then
    echo "Error: --remote is not in the correct format (user@server:directory)."
    exit 1
  fi

  REMOTE_USER="${REMOTE%%@*}"
  REMOTE_SERVER="${REMOTE#*@}"
  REMOTE_SERVER="${REMOTE_SERVER%%:*}"
  REMOTE_DIR="${REMOTE#*:}"
fi
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
  RAWGITSERVER=raw.githubusercontent.com

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

if [[ ! -z "$REMOTE" ]]; then
  REMOTE_USER_SERVER=${REMOTE_USER}@${REMOTE_SERVER}
  REMOTE_TASK_DIR=$REMOTE_DIR/logs/${TASK}
fi
LOCAL_TASK_DIR="$LOCAL_AMP_DIR"/logs/${TASK}
LAST_CHECKPOINT=
while true;
do
  if [[ ! -z "$REMOTE" ]]; then
    echo Fetching checkpoint from ${REMOTE_USER_SERVER}
    LAST_CHECKPOINT_DIR=$(ssh ${REMOTE_USER_SERVER} 'find '$REMOTE_TASK_DIR' -type d -exec stat --format="%Y %n" {} \; | sort -n | tail -1 | cut -d" " -f2')
    LATEST_CHECKPOINT=
    if [ ! -z "$LAST_CHECKPOINT_DIR" ]; then
      LATEST_CHECKPOINT=$(ssh ${REMOTE_USER_SERVER} "find '$LAST_CHECKPOINT_DIR' -name '*.pt' -type f -exec stat --format='%Y %n' {} \; | sort -n | tail -1 | cut -d' ' -f2")
    fi
  else
    LAST_CHECKPOINT_DIR=$(find "$LOCAL_TASK_DIR" -type d -exec stat --format="%Y %n" {} \; | sort -n | tail -1 | cut -d" " -f2)
    LATEST_CHECKPOINT=
    if [ ! -z "$LAST_CHECKPOINT_DIR" ]; then
      LATEST_CHECKPOINT=$(find "$LAST_CHECKPOINT_DIR" -name '*.pt' -type f -exec stat --format='%Y %n' {} \; | sort -n | tail -1 | cut -d' ' -f2)
    fi
  fi
  echo LAST_CHECKPOINT_DIR: $LAST_CHECKPOINT_DIR
  echo LATEST_CHECKPOINT: $LATEST_CHECKPOINT
  echo LAST_CHECKPOINT: $LAST_CHECKPOINT
  if [ ! -z "$LATEST_CHECKPOINT" ]; then
    # latest checkpoint filename
    CHECKPOINT=$(basename $LATEST_CHECKPOINT)
    LOCAL_CHECKPOINT_DIR=$(basename "$LAST_CHECKPOINT_DIR")
    LOCAL_TASK_CHECKPOINT_DIR="$LOCAL_TASK_DIR/$LOCAL_CHECKPOINT_DIR/"
    HISTORY_DIR="$LOCAL_AMP_DIR"/history/$TASK/$LOCAL_CHECKPOINT_DIR
    echo Latest checkpoint: $CHECKPOINT from ${REMOTE_USER_SERVER}

    if [[ ! -f "$LOCAL_TASK_DIR"/"$LATEST_CHECKPOINT" || "$LAST_CHECKPOINT" != "$CHECKPOINT" ]]; then
      if [[ ! -z "$REMOTE" ]]; then
        rsync -avz $REMOTE_USER_SERVER:"$LATEST_CHECKPOINT" "$LOCAL_TASK_DIR/$(basename "$LAST_CHECKPOINT_DIR")/" || exit
      fi

      if [ ! -z "$LOCAL_TASK_REPO" ]; then
        cd "$LOCAL_TASK_REPO" || exit
        git reset HEAD .
        git restore .
        git clean -fd
        git checkout main || exit
        VIDEO_OUTPUT_FILE=$LOCAL_TASK_REPO/$VIDEO_FILE
        SIM2SIM_VIDEO_OUTPUT_FILE=$LOCAL_TASK_REPO/$SIM2SIM_VIDEO_FILE
      else
        SIM2SIM_VIDEO_OUTPUT_FILE=$SIM2SIM_VIDEO_FILE
      fi

      cd "$LOCAL_AMP_DIR" || exit
      GYM_PLOT_COMMAND_ACTION=1 python legged_gym/scripts/record_policy.py --task=$TASK || exit
      python plot_action_obs.py -png --width 1600 --height 1080

      days=0
      hours=0
      minutes=0
      CHECKPOINT_NUMBER="${CHECKPOINT#model_}"
      CHECKPOINT_NUMBER="${CHECKPOINT_NUMBER%.*}"
      if [ "$CHECKPOINT_NUMBER" != "0" ]; then
        mtime1=$(find "$LOCAL_TASK_CHECKPOINT_DIR" -type f -printf '%T@ %p\n' | sort -n | head -n 1 | awk '{print int($1)}')
        mtime2=$(stat -c %Y "$LOCAL_TASK_CHECKPOINT_DIR"/"$CHECKPOINT")
        diff_seconds=$((mtime2 - mtime1))
        days=$((diff_seconds / 86400))
        hours=$(( (diff_seconds % 86400) / 3600 ))
        minutes=$(( (diff_seconds % 3600) / 60 ))
      fi
      CAPTION="${days}d ${hours}h ${minutes}m"
      echo CAPTION: $CAPTION
      ffmpeg -y -i record.mp4 -vf "drawtext=text='$CAPTION':fontcolor=white:fontsize=56:x=(w-text_w)/2:y=h-(text_h*3):alpha='if(lt(t,1),0,if(lt(t,2),t-1,if(lt(t,5),1,if(lt(t,6),1-(t-5),0))))'" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart "$VIDEO_OUTPUT_FILE" || exit
      mkdir -p "$HISTORY_DIR"
      rsync -auv "$VIDEO_OUTPUT_FILE" "$HISTORY_DIR"/video_${CHECKPOINT_NUMBER}.mp4
      cp -f "$LOCAL_TASK_DIR"/exported/policies/model.pt "$HISTORY_DIR"/model_${CHECKPOINT_NUMBER}.pt || exit
      cp -f "$LOCAL_TASK_DIR"/exported/policies/policy.onnx "$HISTORY_DIR"/policy_${CHECKPOINT_NUMBER}.onnx || exit

      fetch_tensordata json "Episode/rew_action_rate" Episode_rew_action_rate
      fetch_tensordata json "Episode/rew_feet_air_time" Episode_rew_feet_air_time
      fetch_tensordata json "Episode/rew_tracking_ang_vel" Episode_rew_tracking_ang_vel
      fetch_tensordata json "Episode/rew_tracking_lin_vel" Episode_rew_tracking_lin_vel
      fetch_tensordata json "Loss/AMP" Loss_AMP
      fetch_tensordata json "Loss/AMP_grad" Loss_AMP_grad
      fetch_tensordata json "Loss/learning_rate" Loss_learning_rate
      fetch_tensordata json "Loss/surrogate" Loss_surrogate
      fetch_tensordata json "Loss/value_function" Loss_value_function
      fetch_tensordata json "Perf/collection%20time" Perf_collection_time
      fetch_tensordata json "Perf/learning_time" Perf_learning_time
      fetch_tensordata json "Perf/total_fps" Perf_total_fps
      fetch_tensordata json "Train/mean_episode_length" Train_mean_episode_length
      fetch_tensordata json "Train/mean_episode_length/time" Train_mean_episode_length_time
      fetch_tensordata json "Train/mean_reward" Train_mean_reward
      fetch_tensordata json "Train/mean_reward/time" Train_mean_reward_time
      python plot_tensorboard.py -png --width 1600 --height 1080 "$HISTORY_DIR"/reports

      echo SIM2SIM_VIDEO_OUTPUT_FILE: $SIM2SIM_VIDEO_OUTPUT_FILE
      python onnx_AMP_mujoco.py -o "$LOCAL_TASK_DIR"/exported/policies/policy.onnx --width=1600 --height=900 --video="sim2sim.mp4" --hide-menu --duration=8
      ffmpeg -y -i sim2sim.mp4 -vf "drawtext=text='Mujoco':fontcolor=white:fontsize=56:x=(w-text_w)/2:y=h-(text_h*3):alpha='if(lt(t,1),0,if(lt(t,2),t-1,if(lt(t,5),1,if(lt(t,6),1-(t-5),0))))'" -c:v libx264 -b:v 128k -crf 23 -preset medium -c:a aac -movflags +faststart "$SIM2SIM_VIDEO_OUTPUT_FILE" || exit

      if [ ! -z "$LOCAL_TASK_REPO" ]; then
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
        cp -f "$LOCAL_AMP_DIR"/plot.png . || exit
        cp -f "$LOCAL_AMP_DIR"/tensorboard.png . || exit
        cp -f "$LOCAL_TASK_DIR"/exported/policies/policy.onnx . || exit
        git add "$VIDEO_FILE" || exit
        git add "$SIM2SIM_VIDEO_FILE" || exit
        git add tensorboard.png || exit
        git add plot.png || exit
        git add policy.onnx || exit
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
          current_datetime=$(date +"%m-%d %H:%M:%S")
          git checkout gh-pages || exit
          sed -i "s|<title>.*</title>|<title>$TASK_NAME</title>|" index.html || exit
          sed -i "s|<h1>.*</h1>|<h1>$TASK_NAME</h1>|" index.html || exit
          sed -i "s|<h2>.*</h2>|<h2>Checkpoint ${CHECKPOINT_NUMBER} ($current_datetime)</h2>|" index.html || exit
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
      fi
      LAST_CHECKPOINT=$CHECKPOINT
    fi
  else
    echo No checkpoint
  fi
  # Check if INTERVAL_MINUTES is equal to 0
  if [ "$INTERVAL_MINUTES" -eq 0 ]; then
    echo Success
    exit 0
  fi
  if [[ ! -z "$REMOTE" ]]; then
    echo Waiting $INTERVAL_MINUTES minutes ...
    sleep "$INTERVAL_SECONDS"
  else
    monitor_new_model_files "$LOCAL_TASK_DIR/$LOCAL_CHECKPOINT_DIR/"
  fi
done
