from manim import *
import numpy as np

config.background_color = WHITE


class FlowOnly(Scene):
    def construct(self):
        # No title - slide header handles it

        # Body ellipse - larger
        body = Ellipse(width=8, height=5.5, color="#bdc3c7", stroke_width=2, fill_opacity=0.02)
        body.move_to(DOWN * 0.3)

        # Anchors - slightly larger
        anch = VGroup(*[
            Triangle(fill_color="#e67e22", fill_opacity=0.9, color="#e67e22", stroke_width=0.5)
            .scale(0.14).move_to([4.3 * np.cos(a * PI / 180), -0.3 + 3.0 * np.sin(a * PI / 180), 0])
            for a in [40, 160, 280, 80, 200, 320, 120, 240, 0]
        ])

        true_pos = np.array([0.5, -0.3, 0])
        # Red target - no label to avoid clutter
        capsule = Dot(true_pos, radius=0.2, color="#e74c3c", fill_opacity=1)

        self.play(Create(body), FadeIn(anch), FadeIn(capsule), run_time=0.5)

        # RF signal lines
        rf = VGroup(*[
            DashedLine(a.get_center(), true_pos, color="#3498db", stroke_width=1,
                      dash_length=0.08, stroke_opacity=0.3)
            for a in anch
        ])
        rf_label = Text("RF measurements (y) condition the velocity field",
                        font_size=20, color="#3498db", weight=BOLD)
        rf_label.move_to(DOWN * 3.5)
        self.play(Create(rf), Write(rf_label), run_time=0.5)
        self.wait(0.3)
        self.play(rf.animate.set_opacity(0.1), FadeOut(rf_label), run_time=0.3)

        # Step 1: Prior samples
        step = Text("t = 0 : sample from cylindrical prior", font_size=22, color="#333333", weight=BOLD)
        step.move_to(DOWN * 3.7)

        np.random.seed(42)
        n = 35
        starts = []
        for _ in range(n):
            while True:
                x = np.random.uniform(-3.5, 3.5)
                y = -0.3 + np.random.uniform(-2.2, 2.2)
                if (x / 3.8) ** 2 + ((y + 0.3) / 2.5) ** 2 < 1:
                    starts.append(np.array([x, y, 0]))
                    break

        particles = [Dot(p, radius=0.07, color="#27ae60", fill_opacity=0.75) for p in starts]
        self.play(Write(step), *[FadeIn(p, scale=0.3) for p in particles], run_time=0.7)
        self.wait(0.4)

        # Step 2: Velocity field - DARKER arrows
        arrows = []
        for gx in np.linspace(-3.0, 3.0, 9):
            for gy in np.linspace(-2.2, 1.8, 7):
                if (gx / 3.5) ** 2 + ((gy + 0.3) / 2.3) ** 2 < 0.85:
                    s = np.array([gx, gy, 0])
                    d = true_pos - s
                    d = d / (np.linalg.norm(d) + 0.01) * 0.45
                    arr = Arrow(s, s + d, color="#4b5563", stroke_width=2.0,
                               tip_length=0.09, max_tip_length_to_length_ratio=0.3)
                    arr.set_opacity(0.55)
                    arrows.append(arr)
        v_grp = VGroup(*arrows)

        step2 = Text("Learned velocity field f(x, t, y)", font_size=22, color="#333333", weight=BOLD)
        step2.move_to(DOWN * 3.5)
        self.play(FadeIn(v_grp), ReplacementTransform(step, step2), run_time=0.5)
        self.wait(0.3)

        # Step 3: ODE integration - particles flow
        ends = [true_pos + np.array([np.random.randn() * 0.3, np.random.randn() * 0.25, 0])
                for _ in range(n)]

        time_steps = [0.2, 0.4, 0.6, 0.8, 1.0]
        for t in time_steps:
            interp = [starts[i] + t * (ends[i] - starts[i]) for i in range(n)]
            new_step = Text(f"t = {t:.1f}  integrating ODE", font_size=22, color="#333333", weight=BOLD)
            new_step.move_to(DOWN * 3.7)

            self.play(
                *[p.animate.move_to(pos) for p, pos in zip(particles, interp)],
                ReplacementTransform(step2, new_step),
                run_time=0.55, rate_func=smooth
            )
            step2 = new_step

        # Fade velocity field
        self.play(FadeOut(v_grp), run_time=0.2)

        # Result: just show convergence
        final = Text("t = 1 : particles converged to posterior p(x|y)",
                     font_size=22, color="#27ae60", weight=BOLD)
        final.move_to(DOWN * 3.7)

        self.play(ReplacementTransform(step2, final), run_time=0.5)

        self.wait(2.5)
