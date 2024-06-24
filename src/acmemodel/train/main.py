from .data import load_raw_data
from .model import get_model
from .train import train


def main(optimize: bool = False) -> str:
    data = load_raw_data()
    model = get_model()
    _, _, log_output = train(model, data, optimize=optimize)
    id = log_output.report['estimator']['id']
    print(f'Trained model id: {id}')

    return id


if __name__ == '__main__':
    main()
