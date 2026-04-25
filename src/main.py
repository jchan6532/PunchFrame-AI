from pose_extractor import draw_pose_on_video


def main():
    input_video = "data/raw_videos/test.mp4"
    output_video = "outputs/labeled_videos/test_pose_output.mp4"

    draw_pose_on_video(input_video, output_video)


if __name__ == "__main__":
    main()