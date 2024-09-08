import torch
from tqdm import tqdm


def train_test_loop(model, train_loader, test_loader, criterion, device, num_epochs=10, learning_rate=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_batches = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({'loss': train_loss / train_batches})

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Testing loop
        model.eval()
        test_loss = 0.0
        test_batches = 0

        with torch.inference_mode():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Test]')
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_batches += 1
                test_pbar.set_postfix({'loss': test_loss / test_batches})

        avg_test_loss = test_loss / test_batches
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Test Loss: {avg_test_loss:.4f}')

    return train_losses, test_losses