for history in 0.0 # 0.125 0.25 0.375 0.5
do 
	for down_sum in 1.0 2.0 3.0 4.0 5.0
	do
		for late_exp in 1.0 1.125 1.25 1.375 1.5 
		do
			for up_sum in 4.0 6.0 8.0 10.0 
			do
				for on_time_exp in 1.0 1.125 1.25 
				do
					echo ${down_sum}-${late_exp}_${up_sum}-${on_time_exp}_${history}
					sudo ../build/client_diff localhost 10001 ../simulation/wifi_walking2.txt /home/vsa/video_lcp/videos/intersection_10m.mp4 0 $down_sum $late_exp $up_sum $on_time_exp $history > ${down_sum}-${late_exp}_${up_sum}-${on_time_exp}_${history}.txt
				done 
			done
		done
	done
done
