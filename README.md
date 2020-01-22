# video_lcp
A latency control protocol for video analytics pipeline

# setup 
darknet should be installed in /build/
cfg files from /cfg/ should be copied to /build/darknet/cfg/

# usage
Server : Server should be started on localhost or on external server 
./server {port_no}
Monitor (optional): Monitor should be started on localhost
./monitor {port_no}
Client : Client should be started on local host, to run without monitor select 0 for monitor port 
./client {server_ip} {server_port} {monitor_port} {video_file}

