# Note on the VAEE posterior factorisation: structured → mean-field

**To:** Gabriele
**From:** Francesco
**Re:** the gating derivation (Appendix A.1) and the ELBO in the Methods chapter

## Summary

While writing up the VAEE methods and the gating derivation, I changed the
variational posterior from the **structured** form used in
`probabilistic_formulation_v7.tex` to a **mean-field** form. The change is
mathematically minor — it lands on the *same* gate $\alpha_k$ and the *same*
training loss — but it is more faithful to what the implementation actually does
and it makes the derivation cleaner. I'd like you to sanity-check it, and in
particular bless the one step noted at the end.

## The two factorisations

The latents are the mask $c$ and the embeddings $z$; the generative flow is
$c \to z \to x$, i.e. $p(c, z, x) = p(c)\,p(z \mid c)\,p(x \mid z)$. The
approximate posterior can be factorised in two natural ways:

- **Structured** (the formulation note, App. A.1/A.2):
  $q(c, z \mid x) = q(z \mid x)\,q(c \mid z)$ — infer $z$ from $x$, then $c$ from $z$.
- **Mean-field** (what I adopted):
  $q(c, z \mid x) = q(z \mid x)\,q(c \mid x)$ — $c$ and $z$ conditionally
  independent given $x$.

## Why mean-field is the faithful description

In the code, the gate is
$$\alpha_k = \mathrm{sigmoid}\!\big(\mathrm{sim}(\mu_k(x), h_k)/\tau\big),$$
and it uses the encoder **mean** $\mu_k(x)$ — *not* the sampled $z_k$. So $\alpha$
is independent of the reparameterisation noise given $x$. That is exactly
$q(c \mid x)$: the mask depends on $x$ only. If the code fed the *sampled* $z_k$
into the similarity, it would be the structured $q(c \mid z)$ — but it doesn't.

## The rewritten Appendix A.1 (mean-field CAVI)

For a mean-field family, the factor maximising the ELBO with the others held
fixed satisfies the standard coordinate-ascent condition
$$\log q^\star(c_k \mid x) = \mathbb{E}_{q(z_k \mid x)}\big[\log p(c_k, z_k)\big] + \text{const}
= \log p(c_k) + \mathbb{E}_{q(z_k \mid x)}\big[\log p(z_k \mid c_k)\big] + \text{const}.$$

The likelihood $p(x \mid z)$ carries no $c_k$ (the flow is $c \to z \to x$), so it
drops into the constant. Taking log-odds of the binary $c_k$:
$$\mathrm{logit}\,\alpha_k = \mathrm{logit}(\pi) + \mathbb{E}_{q(z_k \mid x)}\!\left[\log \frac{p(z_k \mid c_k=1)}{p(z_k \mid c_k=0)}\right].$$

With the Gated Gaussian prior $p(z_k \mid c_k=1) = \mathcal{N}(h_k, \sigma_0^2 I)$,
$p(z_k \mid c_k=0) = \mathcal{N}(0, \sigma_0^2 I)$, the shared covariance makes the
log-likelihood-ratio **linear in $z_k$**:
$$\log \frac{p(z_k \mid c_k=1)}{p(z_k \mid c_k=0)} = \frac{1}{\sigma_0^2}\Big(z_k \cdot h_k - \tfrac{1}{2}\|h_k\|^2\Big).$$

Because it is linear, the expectation over $q(z_k \mid x) = \mathcal{N}(\mu_k, \sigma_0^2 I)$
is **exactly** the substitution $z_k \to \mu_k$ — no approximation. Hence
$$\alpha_k(x) = \mathrm{sigmoid}\!\left(\frac{1}{\sigma_0^2}\Big(\mu_k(x) \cdot h_k - \tfrac{1}{2}\|h_k\|^2\Big) + \mathrm{logit}(\pi)\right),$$
identical to App. A.1's result. The constants are then dropped and $1/\sigma_0^2$
folded into the learnable inverse temperature $1/\tau$, recovering the practical
gate.

## What changes, concretely

1. **The mean substitution is now exact, not a shortcut.** In the original A.1 we
   derived $p(c_k \mid z_k)$ and then "substituted $\mu_k$ for $z_k$", which read
   like an approximation. Under mean-field it is the exact value of the expectation,
   thanks to linearity of the log-odds in $z_k$.

2. **The ELBO terms simplify.** Under mean-field:
   - mask-prior KL $= D_{\mathrm{KL}}(q(c_k \mid x)\,\|\,p(c_k))$ — **no**
     $\mathbb{E}_{q(z_k|x)}$ wrapper;
   - conditional-embedding KL $= \mathbb{E}_{q(c_k \mid x)}[D_{\mathrm{KL}}(q(z_k \mid x)\,\|\,p(z_k \mid c_k))]$,
     which is in closed form directly (the two Gaussians share $\sigma_0^2 I$):
     $$\tfrac{1}{2\sigma_0^2}\big(\alpha_k \|\mu_k - h_k\|^2 + (1-\alpha_k)\|\mu_k\|^2\big).$$
     This is exactly the loss term, with $\tfrac{1}{2\sigma_0^2}$ absorbed into $\gamma$.

3. **The stop-gradient $\mathrm{sg}(\alpha)$ now has a single role.** In the
   structured derivation it did double duty (preventing $\alpha \to 0$ cheating
   *and* "restoring" the closed-form KL). Under mean-field the closed form is
   immediate, so $\mathrm{sg}(\alpha)$ is needed only to stop the network driving
   $\alpha \to 0$ to minimise the embedding penalty irrespective of reconstruction.
   I removed the "restores the closed form" justification from the text.

## The reassuring equivalence (and the one step to bless)

Since $x$ depends on the mask only through $z$, we have $c_k \perp x \mid z_k$, so
the true posterior factor is $p(c_k \mid z_k, x) = p(c_k \mid z_k)$ — itself a
sigmoid of the same linear log-odds. The optimal mean-field gate $\alpha_k(x)$ is
therefore **exactly that true posterior factor evaluated at the embedding mean**
$z_k = \mu_k(x)$. In other words, conditioning the mask on $x$ (through $\mu_k$)
loses nothing relative to conditioning it on a sampled $z_k$ — which is why
gating on the predicted mean is principled, not a convenience.

**The step I'd like you to confirm:** treating $\alpha_k(x)$ — the gate evaluated
at the posterior mean — as the optimal mean-field mask factor (equivalently, the
exact posterior factor at the mean). Everything downstream (the ELBO, the loss,
the experiments) is unchanged; this is about how we *justify* the gate in the
write-up. If you'd rather I keep the structured $q(c \mid z)$ presentation from
the formulation note, that's a small revert — but I think the mean-field version
is both truer to the code and cleaner to read.
