ports="39999 39998 39997 39996"
for port in ${ports}; do
    echo "run redis at localhost:${port}"
    redis-server \
        --daemonize yes \
        --port ${port} \
        --maxclients 100000 \
        --maxmemory 0 \
        --maxmemory-policy noeviction \
        --appendonly no \
        --save "" \
        --protected-mode no
done
