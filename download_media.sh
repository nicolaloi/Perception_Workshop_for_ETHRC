pip install gdown -q

current_dir=$(pwd)

function download_with_gdown() {
    local file_id=$1
    local output_file=$2
    local output_path=$3

    gdown ${file_id} -O ${output_file}
    unzip -q ${output_file} -d ${output_path}
    rm -f ${output_file}
}

# https://drive.google.com/file/d/1dSePXG0senGdXkzgwRf1hhvGYUF3Qdxt/view?usp=sharing
download_with_gdown 1dSePXG0senGdXkzgwRf1hhvGYUF3Qdxt videos.zip ${current_dir}/2_Optical_Flow

# https://drive.google.com/file/d/1UUMhB39Wma5sJeL-eUG7LUBtdOBpQcN4/view?usp=sharing
download_with_gdown 1UUMhB39Wma5sJeL-eUG7LUBtdOBpQcN4 images.zip ${current_dir}/3_Visual_Odometry

# https://drive.google.com/file/d/1kVYhBWxh83IqckrPOTZyJ_8zM3M0yEVz/view?usp=sharing
download_with_gdown 1kVYhBWxh83IqckrPOTZyJ_8zM3M0yEVz image_database.zip ${current_dir}/4_Bag_of_Visual_Words

echo "Downloads completed."
