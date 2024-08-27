VIDEO_DIR=$1
find "$VIDEO_DIR" -type f -name "*.mp4" -print0 | xargs -0 ls -1tr > mp4_list.txt
sed -i "s/^/file '/;s/$/'/" mp4_list.txt
ffmpeg -f concat -safe 0 -i mp4_list.txt -c copy output.mp4
#ffmpeg -i output.mp4 -filter:v "setpts=0.25*PTS" -an output_fast2.mp4
#ffmpeg -i output.mp4 -filter:v "setpts=0.125*PTS" -an output_fast4.mp4

#ffmpeg -i output_fast2.mp4 -i MusicaDeCirco-BennyHill.ogg -c:v copy -c:a aac -strict experimental -shortest DuckitySax.mp4