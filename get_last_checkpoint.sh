# last=$(ssh -p4242 apirrone@s-nguyen.net "cd /home/apirrone/MISC/AMP_for_hardware/logs/bdx_amp/*_$1/ ; ls -lt | sed -n '2 p' | grep -io 'model*.pt'")
last=$(ssh -p4242 apirrone@s-nguyen.net "cd /home/apirrone/MISC/AMP_for_hardware/logs/bdx_amp/*_$1/ ; ls *.pt -lt | head -n 1" | awk '{print $8}')
scp -p4242 apirrone@s-nguyen.net:/home/apirrone/MISC/AMP_for_hardware/logs/bdx_amp/*_$1/$last logs/bdx_amp/aze/
