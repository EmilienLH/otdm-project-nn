function accuracy = uo_accuracy(X, y, w)
    predictions = X * w >= 0.5;
    accuracy = sum(predictions == y) / length(y);
end