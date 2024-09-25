import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

def train_test_loop(model, train_loader, test_loader, criterion, device, writer, num_epochs=10, learning_rate=0.001, patience=5):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Define scaler for automatic scaling of gradients
    scaler = GradScaler()

    # early stopping
    best_test_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_batches = 0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Enabling autocast for automatic mixed precision
            with autocast():
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and gradient scaling
            scaler.scale(loss).backward()

            # Update model parameters
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_batches += 1

            train_pbar.set_postfix({'loss': train_loss / train_batches}) # Update the progress bar

            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / train_batches
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)

        # Log training metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Testing loop
        model.eval()
        test_loss = 0.0
        test_batches = 0
        correct_test = 0
        total_test = 0

        with torch.inference_mode():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Test]')
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_batches += 1
                test_pbar.set_postfix({'loss': test_loss / test_batches})

                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        avg_test_loss = test_loss / test_batches
        test_accuracy = correct_test / total_test
        test_losses.append(avg_test_loss)

        # Log testing metrics
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        # Display the epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
        print(f'  Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}')

        # Early stopping check
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            model.load_state_dict(best_model)
            break

    return train_losses, test_losses