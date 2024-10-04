function accuracy = uo_nn_accuracy(X, y, w)
    % Define activation function
    sig = @(X) 1 ./ (1 + exp(-X));
    y_pred = sig(w' * sig(X));
    
    % Calculate accuracy
    predictions = round(y_pred);
    accuracy = sum(predictions == y) / length(y);
end