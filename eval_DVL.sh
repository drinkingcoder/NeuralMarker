# pretrained model check
# echo "===> Checking pretrained model"
# if [ ! -f "~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth" ];
# then
#     cp ./pre_trained_model/backbone_weight/resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints/
# fi
# if [ ! -f "~/.cache/torch/hub/checkpoints/twins_svt_large-90f6aaa9.pth" ];
# then
#     cp ./pre_trained_model/backbone_weight/twins_svt_large-90f6aaa9.pth ~/.cache/torch/hub/checkpoints/
# fi
# if [ ! -f "~/.cache/torch/hub/checkpoints/vgg16-397923af.pth" ];
# then
#     cp ./pre_trained_model/backbone_weight/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/
# fi

root="./data/DVL"
save_root="./output"

blend_type="L"
blend_method="twins-onestage"
img_num=10
scn_num=10
start_img_id=0
start_scene_id=0
source_id=1

# kornia version set
echo "===> Checking kornia version"
version=$(eval echo $(pip show kornia |grep Version |cut -d ":" -f2))
echo "kornia version: ${version}"
if [ "$blend_method" == "ransac-flow" ];
then
    echo "${blend_method} requies kornia version: 0.1.4.post2"
    if [ "$version" != "0.1.4.post2" ];
    then
        pip install --upgrade kornia==0.1.4.post2
    fi
else
    echo "${blend_method} requies kornia version: 0.6.3"
    if [ "$version" != "0.6.3" ];
    then
        pip install --upgrade kornia==0.6.3
    fi
fi

python evaluation_DVL.py --root $root --save_root $save_root \
 --blend_type $blend_type --blend_method $blend_method \
 --img_num $img_num --scn_num $scn_num --start_img_id $start_img_id --start_scene_id $start_scene_id --source_id $source_id \
 --with_mask --use_colormap --save
