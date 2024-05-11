import pickle
import numpy as np
import matplotlib.pyplot as plt


def calc_confidence_avg(confidences, threshold=20):
    confidence_avg = []
    cnt = []
    for i in range(len(confidences)):
        for j in range(len(confidences[i])):
            if len(confidence_avg) < j+1:
                confidence_avg.append(confidences[i][j])
                cnt.append(1)
            else:
                confidence_avg[j] = confidence_avg[j] + confidences[i][j]
                cnt[j] = cnt[j] + 1
    
    for i in range(len(confidence_avg)):
        if cnt[i] < threshold:
            confidence_avg = confidence_avg[:i-1]
        else:
            confidence_avg[i] = confidence_avg[i] / cnt[i]

    return confidence_avg


if __name__=="__main__":
    file_path_past = "/home/4/ud02274/navigation/myss/sound-spaces/data/models/ss1-savi/mp3d/ss1savi-savi-iprl-ft-past-obsenc-offplicy-tf-duration/video_dir/confidences.pickle"
    with open(file_path_past, mode="rb") as f:
        confidences_past = pickle.load(f)

    file_path_future = "/home/4/ud02274/navigation/myss/sound-spaces/data/models/ss1-savi/mp3d/ss1savi-savi-iprl-ft-future-obsenc-offplicy-tf-duration/video_dir/confidences.pickle"
    with open(file_path_future, mode="rb") as f:
        confidences_future = pickle.load(f)

    confidence_avg_past = calc_confidence_avg(confidences_past)
    # print(confidence_avg_past)

    confidence_avg_future = calc_confidence_avg(confidences_future)
    # print(confidence_avg_future)

    plt.plot(confidence_avg_past, label="past")
    plt.plot(confidence_avg_future, label="future")
    plt.plot(
        [np.mean(confidence_avg_past) for _ in range(max(len(confidence_avg_past), len(confidence_avg_future)))],
        label="past avg", 
    )
    plt.plot(
        [np.mean(confidence_avg_future) for _ in range(max(len(confidence_avg_past), len(confidence_avg_future)))],
        label="future avg", 
    )

    plt.xlabel("Step")
    plt.ylabel("Confidence")

    plt.legend()
    # Saving the plot as an image
    plt.savefig('data/imgs/confidence.png')
    