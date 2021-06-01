# gcloud auth list
# gcloud config set project <PROJECT_ID>
# gcloud services enable videointelligence.googleapis.com
# mkdir ~/video-intelligence
# export PROJECT_ID=$(gcloud config get-value core/project)
# gcloud iam service-accounts create my-video-intelligence-sa \
# export GOOGLE_APPLICATION_CREDENTIALS=~/video-intelligence/key.json
# cd ~/video-intelligence
# virtualenv venv
# source venv/bin/activate
# pip install ipython google-cloud-videointelligence==2.1.0
# ipython

# Detect shot changes

from google.cloud import videointelligence_v1 as vi
def detect_shot_changes(video_uri):
    video_client = vi.VideoIntelligenceServiceClient()
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=[vi.Feature.SHOT_CHANGE_DETECTION],
    )
    print(f"Processing video: {video_uri}...")
    operation = video_client.annotate_video(request)
    return operation.result()


video_uri = "gs://cloudmleap/video/next/JaneGoodall.mp4"

response = detect_shot_changes(video_uri)


def print_video_shots(response):
    # First result only, as a single video is processed
    shots = response.annotation_results[0].shot_annotations
    print(f" Video shots: {len(shots)} ".center(40, "-"))
    for i, shot in enumerate(shots):
        t1 = shot.start_time_offset.total_seconds()
        t2 = shot.end_time_offset.total_seconds()
        print(f"{i+1:>3} | {t1:7.3f} | {t2:7.3f}")


print_video_shots(response)

#  Detect labels

from google.cloud import videointelligence_v1 as vi


def detect_labels(video_uri, mode, segments=None):
    video_client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.LABEL_DETECTION]
    config = vi.LabelDetectionConfig(label_detection_mode=mode)
    context = vi.VideoContext(segments=segments, label_detection_config=config)
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=features,
        video_context=context,
    )
    print(f"Processing video: {video_uri}...")
    operation = video_client.annotate_video(request)
    return operation.result()


from datetime import timedelta

video_uri = "gs://cloudmleap/video/next/JaneGoodall.mp4"
mode = vi.LabelDetectionMode.SHOT_MODE
segment = vi.VideoSegment(
    start_time_offset=timedelta(seconds=0),
    end_time_offset=timedelta(seconds=37),
)

response = detect_labels(video_uri, mode, [segment])


from google.cloud import videointelligence_v1 as vi


def detect_labels(video_uri, mode, segments=None):
    video_client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.LABEL_DETECTION]
    config = vi.LabelDetectionConfig(label_detection_mode=mode)
    context = vi.VideoContext(segments=segments, label_detection_config=config)
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=features,
        video_context=context,
    )
    print(f"Processing video: {video_uri}...")
    operation = video_client.annotate_video(request)
    return operation.result()


from datetime import timedelta

video_uri = "gs://cloudmleap/video/next/JaneGoodall.mp4"
mode = vi.LabelDetectionMode.SHOT_MODE
segment = vi.VideoSegment(
    start_time_offset=timedelta(seconds=0),
    end_time_offset=timedelta(seconds=37),
)

response = detect_labels(video_uri, mode, [segment])


def print_video_labels(response):
    # First result only, as a single video is processed
    labels = response.annotation_results[0].segment_label_annotations
    sort_by_first_segment_confidence(labels)

    print(f" Video labels: {len(labels)} ".center(80, "-"))
    for label in labels:
        categories = category_entities_to_str(label.category_entities)
        for segment in label.segments:
            confidence = segment.confidence
            t1 = segment.segment.start_time_offset.total_seconds()
            t2 = segment.segment.end_time_offset.total_seconds()
            print(
                f"{confidence:4.0%}",
                f"{t1:7.3f}",
                f"{t2:7.3f}",
                f"{label.entity.description}{categories}",
                sep=" | ",
            )


def sort_by_first_segment_confidence(labels):
    labels.sort(key=lambda label: label.segments[0].confidence, reverse=True)


def category_entities_to_str(category_entities):
    if not category_entities:
        return ""
    entities = ", ".join([e.description for e in category_entities])
    return f" ({entities})"


def print_shot_labels(response):
    # First result only, as a single video is processed
    labels = response.annotation_results[0].shot_label_annotations
    sort_by_first_segment_start_and_confidence(labels)

    print(f" Shot labels: {len(labels)} ".center(80, "-"))
    for label in labels:
        categories = category_entities_to_str(label.category_entities)
        print(f"{label.entity.description}{categories}")
        for segment in label.segments:
            confidence = segment.confidence
            t1 = segment.segment.start_time_offset.total_seconds()
            t2 = segment.segment.end_time_offset.total_seconds()
            print(f"{confidence:4.0%} | {t1:7.3f} | {t2:7.3f}")


def sort_by_first_segment_start_and_confidence(labels):
    def first_segment_start_and_confidence(label):
        first_segment = label.segments[0]
        ms = first_segment.segment.start_time_offset.ToMilliseconds()
        return (ms, -first_segment.confidence)


print_shot_labels(response)

# Detect explicit content

from google.cloud import videointelligence_v1 as vi


def detect_explicit_content(video_uri, segments=None):
    video_client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.EXPLICIT_CONTENT_DETECTION]
    context = vi.VideoContext(segments=segments)
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=features,
        video_context=context,
    )
    print(f"Processing video: {video_uri}...")
    operation = video_client.annotate_video(request)
    return operation.result()


from datetime import timedelta

video_uri = "gs://cloudmleap/video/next/JaneGoodall.mp4"
segment = vi.VideoSegment(
    start_time_offset=timedelta(seconds=0),
    end_time_offset=timedelta(seconds=10),
)

response = detect_explicit_content(video_uri, [segment])

def print_explicit_content(response):
    from collections import Counter

    # First result only, as a single video is processed
    frames = response.annotation_results[0].explicit_annotation.frames
    likelihood_counts = Counter([f.pornography_likelihood for f in frames])

    print(f" Explicit content frames: {len(frames)} ".center(40, "-"))
    for likelihood in vi.Likelihood:
        print(f"{likelihood.name:<22}: {likelihood_counts[likelihood]:>3}")

print_explicit_content(response)

# Transcribe speech

from google.cloud import videointelligence_v1 as vi


def transcribe_speech(video_uri, language_code, segments=None):
    video_client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.SPEECH_TRANSCRIPTION]
    config = vi.SpeechTranscriptionConfig(
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    context = vi.VideoContext(
        segments=segments,
        speech_transcription_config=config,
    )
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=features,
        video_context=context,
    )
    print(f"Processing video: {video_uri}...")
    operation = video_client.annotate_video(request)
    return operation.result()


from google.cloud import videointelligence_v1 as vi


from datetime import timedelta

video_uri = "gs://cloudmleap/video/next/JaneGoodall.mp4"
language_code = "en-GB"
segment = vi.VideoSegment(
    start_time_offset=timedelta(seconds=55),
    end_time_offset=timedelta(seconds=80),
)

response = transcribe_speech(video_uri, language_code, [segment])

def print_video_speech(response, min_confidence=0.8):
    def keep_transcription(transcription):
        return min_confidence <= transcription.alternatives[0].confidence

    # First result only, as a single video is processed
    transcriptions = response.annotation_results[0].speech_transcriptions
    transcriptions = [t for t in transcriptions if keep_transcription(t)]

    print(f" Speech Transcriptions: {len(transcriptions)} ".center(80, "-"))
    for transcription in transcriptions:
        best_alternative = transcription.alternatives[0]
        confidence = best_alternative.confidence
        transcript = best_alternative.transcript
        print(f" {confidence:4.0%} | {transcript.strip()}")

print_video_speech(response)