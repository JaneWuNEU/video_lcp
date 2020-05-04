for down_sum in 1.0 2.5 5.0 7.5
do
  for late_exp in 1.0 1.125 1.25 1.375 1.5 1.625 1.75 
  do
    for up_sum in 2.5 5.0 7.5 10.0 
	do
      for on_time_exp in 1.0 1.125 1.25 
	  do
		echo ${down_sum}-${late_exp}_${up_sum}-${on_time_exp}
        sudo ../build/client_diff localhost 10001 ../simulation/sample.txt ../videos/intersection.mp4 $down_sum $late_exp $up_sum $on_time_exp > ${down_sum}-${late_exp}_${up_sum}-${on_time_exp}.txt
      done 
    done
  done
done