from analysis import generate_data, analyze

if __name__ == "__main__":
    X, y, features = generate_data()
    print("Dataset generated with features:", features)
    print("\nFeature importance interpretation:")
    for line in analyze():
        print(line)
