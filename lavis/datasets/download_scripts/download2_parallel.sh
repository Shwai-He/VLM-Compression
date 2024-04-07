urls=(
  "http://images.cocodataset.org/zips/train2014.zip"
  "http://images.cocodataset.org/zips/val2014.zip"
  "http://images.cocodataset.org/zips/test2014.zip"
  "http://images.cocodataset.org/zips/test2015.zip"
  "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/didemo/didemo_videos.tar.gz"
  "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
  "https://download2295.mediafire.com/4bb7p74xrbgg/x3rrbe4hwp04e6w/train_val_videos.zip"
  "https://download2390.mediafire.com/79hfq3592lqg/czh8sezbo9s4692/test_videos.zip"
  "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"
  "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
  "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
)

target_path=(
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/coco/train2014.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/coco/val2014.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/coco/test2014.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/coco/test2015.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/didemo/didemo_videos.tar.gz"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/gqa/images.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/msrvtt/train_val_videos.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/msrvtt/test_videos.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/msvd/YouTubeClips.tar"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/vg/images.zip"
  "/mnt/petrelfs/dongdaize.d/workspace/sh/data/vg/images2.zip"
)

# 不包含：flickr nocaps sbu vqa

gpu=0
cpu=16
quotatype=auto

for ((i = 0; i < ${#urls[@]}; i++)); do
  OMP_NUM_THREADS=16 srun --partition=MoE --job-name=download --mpi=pmi2 --gres=gpu:${gpu} -n1 --ntasks-per-node=1 -c ${cpu} --kill-on-bad-exit=1 --quotatype=${quotatype} \
    curl -C - --remote-time --fail --create-dirs -o ${target_path[${i}]} ${urls[${i}]} &
  sleep 1
done
wait

# 多跑几轮，从断点重新下载，防止没下载完
for ((i = 0; i < ${#urls[@]}; i++)); do
  OMP_NUM_THREADS=16 srun --partition=MoE --job-name=download --mpi=pmi2 --gres=gpu:${gpu} -n1 --ntasks-per-node=1 -c ${cpu} --kill-on-bad-exit=1 --quotatype=${quotatype} \
    curl -C - --remote-time --fail --create-dirs -o ${target_path[${i}]} ${urls[${i}]} &
  sleep 1
done

wait
for ((i = 0; i < ${#urls[@]}; i++)); do
  OMP_NUM_THREADS=16 srun --partition=MoE --job-name=download --mpi=pmi2 --gres=gpu:${gpu} -n1 --ntasks-per-node=1 -c ${cpu} --kill-on-bad-exit=1 --quotatype=${quotatype} \
    curl -C - --remote-time --fail --create-dirs -o ${target_path[${i}]} ${urls[${i}]} &
  sleep 1
done
wait

for ((i = 0; i < ${#urls[@]}; i++)); do
  OMP_NUM_THREADS=16 srun --partition=MoE --job-name=download --mpi=pmi2 --gres=gpu:${gpu} -n1 --ntasks-per-node=1 -c ${cpu} --kill-on-bad-exit=1 --quotatype=${quotatype} \
    curl -C - --remote-time --fail --create-dirs -o ${target_path[${i}]} ${urls[${i}]} &
  sleep 1
done
wait