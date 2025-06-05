import torch
from common.utils import psnr


def modulation_consistency(modulations, modulations_bootstrapped, bs):
    """
    A function that calculates the L2-distance between the modulations and a bootstrapped target.
    Proposed in 'Learning Large-scale Neural Fields via Context Pruned Meta-Learning' by Jihoon Tack, et al. (2023)

    Everything is implemented to use this bootstrap correction. It is however NOT USED IN OUR PAPER.
    """
    updated_modulation = modulations_bootstrapped - modulations
    updated_modulation = updated_modulation.view(bs, -1)
    modulation_norm = torch.mean(updated_modulation**2, dim=-1)
    return modulation_norm


def get_grad_norm(grads, detach=True):
    grad_norm_list = []
    for grad in grads:
        if grad is None:
            grad_norm = 0
        else:
            if detach:
                grad_norm = torch.norm(grad.data, p=2, keepdim=True).unsqueeze(dim=0)
                print(grad_norm.shape)
            else:
                grad_norm = torch.norm(grad, p=2, keepdim=True).unsqueeze(dim=0)

        grad_norm_list.append(grad_norm)
    print(len(grad_norm_list))
    return torch.norm(torch.cat(grad_norm_list, dim=0), p=2, dim=1)

def get_grad_norm_mlp(grads, detach=True):
    """
    grads : Iterable[Optional[Tensor]]  (e.g. tuple returned by autograd.grad)
    Returns a scalar ≈ ‖grads‖₂  (L2-norm over *all* parameters)
    """
    norms = []
    for g in grads:
        if g is None:
            # Nothing contributed by this parameter
            continue
        tensor = g.detach() if detach else g
        norms.append(torch.norm(tensor, p=2))
    if not norms:                       # all None
        return torch.tensor(0., device='cpu')
    return torch.norm(torch.stack(norms), p=2)


def train_step(
    args, step, model_wrapper, optimizer, data, metric_logger, logger, conditions=None, conditioning_type="mlp"
):
    """
    Function that performs a single meta update
    """
    model_wrapper.model.train()
    model_wrapper.coord_init()  # Reset coordinates
    model_wrapper.model.reset_modulations()  # Reset modulations (zero-initialization)

    batch_size = data.size(0)

    if step % args.print_step == 0:
        conditions_0 = None
        if conditions is not None:
            conditions_learned_init = conditions
            num_conditions = conditions.shape[-1]
            conditions_0 = (
                torch.tensor([0 for i in range(num_conditions)])
                .unsqueeze(0)
                .repeat((batch_size, 1))
                .float()
                .cuda()
            )
            learned_init_0 = model_wrapper(conditions=conditions_0)
        learned_init = model_wrapper(conditions=conditions_learned_init)
        input = data

    #     print("conditions: ", conditions.shape)
    """ Inner-loop optimization for G steps """
    loss_in = inner_adapt(
        model_wrapper=model_wrapper,
        data=data,
        step_size=args.inner_lr,
        num_steps=args.inner_steps,
        first_order=False,
        sample_type=args.sample_type,
        conditions=conditions,
        conditioning_type=conditioning_type
    )

    """ Compute reconstruction loss using full context set"""
    model_wrapper.coord_init()
    modulations = (
        model_wrapper.model.modulations.clone()
    )  # Store modulations for consistency loss (not used)
    loss_out = model_wrapper(data, conditions=conditions)  # Compute reconstruction loss
    if step % args.print_step == 0:
        images = model_wrapper(conditions=conditions_learned_init)  # Sample images

    """ Bootstrap correction for additional steps (NOT USED IN THIS PAPER) """
    _ = inner_adapt(
        model_wrapper=model_wrapper,
        data=data,
        step_size=args.inner_lr_boot,
        num_steps=args.inner_steps_boot,
        first_order=True,
        conditions=conditions,
        conditioning_type=conditioning_type
    )
    modulations_bootstrapped = model_wrapper.model.modulations.detach()
    if step % args.print_step == 0:
        target_boot = model_wrapper(conditions=conditions_learned_init)

    """ Modulation consistency loss and loss aggregation (WE ONLY USE RECONSTRUCTION LOSS) """
    if conditioning_type == "concatenation":
        loss_boot = modulation_consistency(
            modulations, modulations_bootstrapped, bs=batch_size
        )
        loss_boot_weighted = args.lam * loss_boot
        loss = loss_out.mean() + loss_boot_weighted.mean()
    else:
        loss = loss_out.mean()

    """ Meta update (optimize shared weights) """
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()

    """ Track stats"""
    metric_logger.meters["loss_inner"].update(loss_in.mean().item(), n=batch_size)
    metric_logger.meters["loss_outer"].update(loss_out.mean().item(), n=batch_size)
    metric_logger.meters["psnr_inner"].update(psnr(loss_in).mean().item(), n=batch_size)
    metric_logger.meters["psnr_outer"].update(
        psnr(loss_out).mean().item(), n=batch_size
    )
    if conditioning_type == "concatenation":
        metric_logger.meters["loss_boot"].update(
            loss_boot_weighted.mean().item(), n=batch_size
        )
    metric_logger.synchronize_between_processes()

    if step % args.print_step == 0:
        logger.scalar_summary(
            "train/loss_inner", metric_logger.loss_inner.global_avg, step
        )
        logger.scalar_summary(
            "train/loss_outer", metric_logger.loss_outer.global_avg, step
        )
        logger.scalar_summary(
            "train/psnr_inner", metric_logger.psnr_inner.global_avg, step
        )
        logger.scalar_summary(
            "train/psnr_outer", metric_logger.psnr_outer.global_avg, step
        )
        if conditioning_type == "concatenation":
            logger.scalar_summary(
                "train/loss_boot", metric_logger.loss_boot.global_avg, step
            )
        logger.log_image("train/img_in", input, step)
        logger.log_image("train/learninit", learned_init, step)
        logger.log_image("train/img_inner", images, step)
        logger.log_image("train/img_bst", target_boot, step)
        logger.log(
            "[TRAIN] [Step %3d] [LossInner %f] [LossOuter %f] [PSNRInner %.3f] [PSNROuter %.3f]"
            % (
                step,
                metric_logger.loss_inner.global_avg,
                metric_logger.loss_outer.global_avg,
                metric_logger.psnr_inner.global_avg,
                metric_logger.psnr_outer.global_avg,
            )
        )

    metric_logger.reset()


def inner_adapt(
    model_wrapper,
    data,
    step_size=1e-2,
    num_steps=3,
    first_order=False,
    sample_type="none",
    conditions=None,
    conditioning_type="mlp"
):
    loss = 0.0  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != "none":
            model_wrapper.sample_coordinates(
                sample_type, data
            )  # Sample coordinates for the training step
        loss = inner_loop_step(model_wrapper, data, step_size, first_order, conditions, conditioning_type)

    return loss


def inner_loop_step(
    model_wrapper, data, inner_lr=1e-2, first_order=False, conditions=None, conditioning_type="mlp"
):
    batch_size = data.size(0)

    with torch.enable_grad():
        # Recompute modulations each step to maintain gradient connection
        loss = model_wrapper(data, conditions)

        # Compute gradients w.r.t. all parameters involved
        if conditioning_type == "concatenation":
            params = list(model_wrapper.model.mlp.parameters()) + [
                model_wrapper.model.modulations
            ]
        elif conditioning_type == "mlp":
            params = list(model_wrapper.model.mlp.parameters())

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            params,
            create_graph=not first_order,
        )

        # Update all parameters involved (MLP and z_general)
        for param, grad in zip(params, grads):
            param.data -= inner_lr * grad

    return loss


def inner_adapt_test_scale(
    model_wrapper,
    data,
    step_size=1e-2,
    num_steps=3,
    first_order=False,
    sample_type="none",
    scale_type="grad",
    conditions=None,
    conditioning_type="mlp"
):
    loss = 0.0  # Initialize outer_loop loss

    for step_inner in range(num_steps):
        #         print("step_inner: ", step_inner)
        if sample_type != "none":
            model_wrapper.sample_coordinates(sample_type, data)

        loss = inner_loop_step_tt_gradscale(
            model_wrapper, data, step_size, first_order, scale_type, conditions, conditioning_type
        )

    return loss


def inner_loop_step_tt_gradscale(
    model_wrapper,
    data,
    inner_lr=1e-2,
    first_order=False,
    scale_type="grad",
    conditions=None,
    conditioning_type="mlp"
):
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()
    #     print("inner loop: ", data.shape)

    if conditioning_type == "concatenation":
        params = list(model_wrapper.model.mlp.parameters()) + [
            model_wrapper.model.modulations
        ]
    elif conditioning_type == "mlp":
        params = list(model_wrapper.model.mlp.parameters())

    with torch.enable_grad():
        subsample_loss = model_wrapper(data, conditions=conditions)

        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            params,
#             model_wrapper.model.modulations,
            create_graph=False,
            allow_unused=True,
        )

        if conditioning_type == "concatenation":
            subsample_grad = subsample_grad[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data, conditions=conditions)

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            params,
#             model_wrapper.model.modulations,
            create_graph=not first_order,
            allow_unused=True,
        )

        if conditioning_type == "concatenation":
            grads = grads[0]

    if scale_type == "grad":
        # Gradient rescaling at test-time
        if conditioning_type == "concatenation":
            subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
            grad_norm = get_grad_norm(grads, detach=True)
            grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
            grad_scale_ = grad_scale.view(
                (batch_size,) + (1,) * (len(grads.shape) - 1)
            ).detach()
        elif conditioning_type == "mlp":
            subsample_grad_norm = get_grad_norm_mlp(subsample_grad, detach=True)  # scalar
            grad_norm             = get_grad_norm_mlp(grads,          detach=True)  # scalar
            grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
            grad_scale = grad_scale.detach()
    else:
        raise NotImplementedError()

    # Update all parameters involved (MLP and z_general)
#     for param, grad in zip(params, grads):
#         param.data -= inner_lr * grad_scale_.mean() * grads

    if conditioning_type == "concatenation":
        model_wrapper.model.modulations = (
            model_wrapper.model.modulations - inner_lr * grad_scale_ * grads
        )

    elif conditioning_type == "mlp":
        with torch.no_grad():
            for p, g in zip(model_wrapper.model.mlp.parameters(), grads):
                if g is None:
                    continue
                p.data.add_(-inner_lr * grad_scale * g)
#         scaled_grads = [grad * grad_scale_.mean() for grad in grads]
#         for param, scaled_grad in zip(params, scaled_grads):
#             param -= inner_lr * scaled_grad

    return loss
