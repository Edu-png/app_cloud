from fastapi import FastAPI
from starlette.responses import StreamingResponse

app = FastAPI()

@app.get("/iris")

def get_iris():
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.decomposition import PCA

    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    # gráfico 2D
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Set1, edgecolors="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title("Iris dataset (2D)")
    plt.show()

    # PCA 3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    X_reduced = PCA(n_components=3).fit_transform(iris.data)

    ax.scatter(
        X_reduced[:,0],
        X_reduced[:,1],
        X_reduced[:,2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolors="k"
    )

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.set_ylabel("2nd eigenvector")
    ax.set_zlabel("3rd eigenvector")

    fig.savefig('iris.png')
    file = open('iris.png', mode = "rg")

    return StreamingResponse(file, media_type = "image/png")