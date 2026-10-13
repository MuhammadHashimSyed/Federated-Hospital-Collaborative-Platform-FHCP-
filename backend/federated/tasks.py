import os
import torch
from celery import shared_task
from django.conf import settings
from .models import TrainingRound, LocalWeightSubmission, AuditLog

@shared_task
def aggregate_weights_for_round(round_id):
    try:
        training_round = TrainingRound.objects.get(id=round_id)
        if training_round.status != 'IN_PROGRESS':
            return
        
        training_round.status = 'AGGREGATING'
        training_round.save()
        
        submissions = LocalWeightSubmission.objects.filter(training_round=training_round)
        if not submissions.exists():
            training_round.status = 'PENDING'
            training_round.save()
            return
            
        # FedAvg Algorithm
        global_state_dict = None
        num_submissions = submissions.count()
        
        for sub in submissions:
            local_weights = torch.load(sub.local_model_file.path, map_location='cpu', weights_only=True)
            
            if global_state_dict is None:
                global_state_dict = {k: v / num_submissions for k, v in local_weights.items()}
            else:
                for k in global_state_dict.keys():
                    global_state_dict[k] += local_weights[k] / num_submissions
                    
        # Save new global model
        global_model_filename = f'global_model_round_{training_round.round_number}.pth'
        global_model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'global', global_model_filename)
        
        os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
        torch.save(global_state_dict, global_model_path)
        
        training_round.global_model_file.name = f'models/global/{global_model_filename}'
        
        # Calculate new global mean AUROC proxy
        mean_auroc = sum(sub.local_mean_auroc for sub in submissions) / num_submissions
        training_round.global_mean_auroc = mean_auroc
        
        training_round.status = 'COMPLETED'
        from django.utils import timezone
        training_round.completed_at = timezone.now()
        training_round.save()
        
        AuditLog.objects.create(
            action=f"Aggregated weights and completed Round {training_round.round_number}"
        )
        return f"Round {training_round.round_number} aggregated successfully."
        
    except Exception as e:
        print(f"Error in aggregate_weights_for_round: {str(e)}")
