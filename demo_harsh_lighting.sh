# check kornia version
echo "===> Checking kornia version"
version=$(eval echo $(pip show kornia |grep Version |cut -d ":" -f2))
echo "kornia version: ${version}"
echo "NeuralMarker requies kornia version: 0.6.3"
if [ "$version" != "0.6.3" ];
then
    pip install --upgrade kornia==0.6.3
fi

exp_dir="/mnt/nas_8/group/weihong/OpticalFlow/NeuralMarkerEval&Demo/demo/demo_harsh_lighting"
marker_name="fantastic_beast.jpg"
scene_name="scene_beast.jpg"
source_name="doctor_strange.jpg"

# marker_name="cards_pic.jpg"
# scene_name="cards.jpg"
# source_name="Coca-Cola-Logo.jpg"

# marker_name="sonic.jpg"
# scene_name="scene_sonic.jpg"
# source_name="fantastic_beast.jpg"


echo "demo: replace marker in harsh lighting environment"
python demo_harsh_lighting.py --exp_dir $exp_dir --marker_name $marker_name \
 --scene_name $scene_name --source_name $source_name 