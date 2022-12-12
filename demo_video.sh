# check kornia version
echo "===> Checking kornia version"
version=$(eval echo $(pip show kornia |grep Version |cut -d ":" -f2))
echo "kornia version: ${version}"
echo "NeuralMarker requies kornia version: 0.6.3"
if [ "$version" != "0.6.3" ];
then
    pip install --upgrade kornia==0.6.3
fi

demo_root="./demo_video"
marker_name="fantastic_beast.jpg"
scene_name="scene.mp4"
movie_name="movie.mp4"
save_name="out.avi"
movie_start_idx=0
scene_start_idx=150

echo "demo: replace marker in scene video with a stream of movie"
python demo_video.py --demo_root $demo_root --marker_name $marker_name \
 --scene_name $scene_name --movie_name $movie_name --save_name $save_name \
 --movie_start_idx $movie_start_idx --scene_start_idx $scene_start_idx

# python demo_video.py --test --save