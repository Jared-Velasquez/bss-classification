import argparse
import sys

def pipeline(args) -> None:
    # Data preprocessing
    # Semantic segmentation of damage buildings to extract damage features
    # Canonicalize descriptions of visual damage with WordNet and NLP
    # Vision and language information synthesis in a scene graph
    # Classify building safety categories according to ATC-20 guidelines

    pass

def main(args):
    parser = argparse.ArgumentParser(
        description="This script is used to analyze PyFG datasets for the connectivity between landmarks and agents."
    )
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="filepath of directory that contains PyFG files to analyze connectivity from",
    )

    args = parser.parse_args()
    pipeline(args)

if __name__ == "__main__":
    main(sys.argv[1:])