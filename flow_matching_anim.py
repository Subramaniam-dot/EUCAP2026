from manim import *
import numpy as np

config.background_color = WHITE


class FlowVsRegression(Scene):
    def construct(self):
        # ======== LAYOUT ========
        divider = Line(UP * 4, DOWN * 4, color="#dddddd", stroke_width=2)

        t_left = Text("Traditional Approaches", font_size=32, color="#c0392b", weight=BOLD)
        t_left.move_to(LEFT * 3.5 + UP * 3.5)
        t_right = Text("Flow Matching (Ours)", font_size=32, color="#27ae60", weight=BOLD)
        t_right.move_to(RIGHT * 3.5 + UP * 3.5)

        self.play(Create(divider), Write(t_left), Write(t_right), run_time=0.7)

        # ======== BODY + ANCHORS (both sides) ========
        def make_scene(cx):
            body = Ellipse(width=4.5, height=3.2, color="#bdc3c7", stroke_width=2, fill_opacity=0.02)
            body.move_to([cx, -0.2, 0])
            angles = [40, 160, 280, 80, 200, 320, 120, 240, 0]
            anch = VGroup(*[
                Triangle(fill_color="#e67e22", fill_opacity=0.9, color="#e67e22", stroke_width=0.5)
                .scale(0.1).move_to([cx + 2.5 * np.cos(a * PI / 180), -0.2 + 1.8 * np.sin(a * PI / 180), 0])
                for a in angles
            ])
            true = np.array([cx + 0.4, -0.3, 0])
            cap = Dot(true, radius=0.1, color="#e74c3c", fill_opacity=1)
            return body, anch, cap, true

        body_l, anch_l, cap_l, true_l = make_scene(-3.5)
        body_r, anch_r, cap_r, true_r = make_scene(3.5)

        self.play(
            Create(body_l), Create(body_r),
            FadeIn(anch_l), FadeIn(anch_r),
            FadeIn(cap_l), FadeIn(cap_r),
            run_time=0.6
        )

        # RF lines
        for anchors, true in [(anch_l, true_l), (anch_r, true_r)]:
            lines = VGroup(*[
                DashedLine(a.get_center(), true, color="#3498db", stroke_width=0.8,
                          dash_length=0.06, stroke_opacity=0.25)
                for a in anchors
            ])
            self.add(lines)

        self.wait(0.3)

        # ================================================================
        # LEFT SIDE: Show 3 traditional methods sequentially
        # ================================================================

        # --- Method 1: Trilateration ---
        m1_title = Text("1. Trilateration", font_size=20, color="#8e44ad", weight=BOLD)
        m1_title.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(Write(m1_title), run_time=0.3)

        # Draw range circles from 3 anchors
        circles = []
        for i in [0, 2, 4]:
            a_pos = anch_l[i].get_center()
            r = np.linalg.norm(a_pos - true_l) + np.random.uniform(-0.3, 0.3)
            c = Circle(radius=r, color="#8e44ad", stroke_width=1.5, stroke_opacity=0.4)
            c.move_to(a_pos)
            circles.append(c)
        circ_grp = VGroup(*circles)
        self.play(Create(circ_grp), run_time=0.6)

        # Intersection point (noisy)
        tri_pred = Dot(true_l + np.array([0.3, -0.4, 0]), radius=0.08, color="#8e44ad")
        self.play(FadeIn(tri_pred), run_time=0.3)

        # Problem text
        prob1 = Text("Needs ≥3 ranges\nSensitive to noise", font_size=13, color="#8e44ad")
        prob1.move_to(LEFT * 5.2 + DOWN * 2.2)
        self.play(Write(prob1), run_time=0.3)
        self.wait(0.5)

        # Fade method 1
        self.play(FadeOut(circ_grp), FadeOut(tri_pred), FadeOut(m1_title), FadeOut(prob1), run_time=0.3)

        # --- Method 2: Fingerprinting (KNN/SVM) ---
        m2_title = Text("2. Fingerprinting (KNN)", font_size=20, color="#2980b9", weight=BOLD)
        m2_title.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(Write(m2_title), run_time=0.3)

        # Show training grid points
        np.random.seed(77)
        grid_dots = []
        for _ in range(15):
            x = -3.5 + np.random.uniform(-1.8, 1.8)
            y = -0.2 + np.random.uniform(-1.2, 1.2)
            if ((x + 3.5) / 2.0) ** 2 + ((y + 0.2) / 1.4) ** 2 < 1:
                d = Dot([x, y, 0], radius=0.05, color="#2980b9", fill_opacity=0.4)
                grid_dots.append(d)
        grid_grp = VGroup(*grid_dots)
        db_label = Text("Training DB", font_size=12, color="#2980b9")
        db_label.move_to(LEFT * 5.2 + UP * 1.5)
        self.play(FadeIn(grid_grp), Write(db_label), run_time=0.4)

        # Highlight nearest neighbors
        nn_lines = VGroup(*[
            Line(grid_dots[i].get_center(), true_l, color="#2980b9", stroke_width=1.5, stroke_opacity=0.5)
            for i in range(min(3, len(grid_dots)))
        ])
        knn_pred = Dot(true_l + np.array([-0.35, 0.25, 0]), radius=0.08, color="#2980b9")
        self.play(Create(nn_lines), FadeIn(knn_pred), run_time=0.4)

        prob2 = Text("Discrete grid\nNo interpolation", font_size=13, color="#2980b9")
        prob2.move_to(LEFT * 5.2 + DOWN * 2.2)
        self.play(Write(prob2), run_time=0.3)
        self.wait(0.5)

        self.play(FadeOut(grid_grp), FadeOut(nn_lines), FadeOut(knn_pred),
                 FadeOut(m2_title), FadeOut(prob2), FadeOut(db_label), run_time=0.3)

        # --- Method 3: Tree-Based (XGBoost) ---
        m3_title = Text("3. Tree-Based (XGBoost)", font_size=20, color="#e67e22", weight=BOLD)
        m3_title.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(Write(m3_title), run_time=0.3)

        # Tree splitting visualization
        tree_lines = VGroup(
            Line(LEFT * 4.8 + UP * 1.2, LEFT * 4.8 + DOWN * 1.2, color="#e67e22", stroke_width=1.5),
            Line(LEFT * 5.5 + UP * 0.3, LEFT * 4.1 + UP * 0.3, color="#e67e22", stroke_width=1.5),
            Line(LEFT * 3.2 + UP * 1.2, LEFT * 3.2 + DOWN * 0.5, color="#e67e22", stroke_width=1.5, stroke_opacity=0.5),
            Line(LEFT * 4.5 + DOWN * 0.5, LEFT * 2.5 + DOWN * 0.5, color="#e67e22", stroke_width=1.5, stroke_opacity=0.5),
        )
        self.play(Create(tree_lines), run_time=0.4)

        # Prediction in a grid cell
        tree_pred = Dot(true_l + np.array([0.25, -0.35, 0]), radius=0.08, color="#e67e22")
        self.play(FadeIn(tree_pred), run_time=0.3)

        prob3t = Text("Axis-aligned splits\nNo smooth interpolation", font_size=13, color="#e67e22")
        prob3t.move_to(LEFT * 3.5 + DOWN * 2.2)
        self.play(Write(prob3t), run_time=0.3)
        self.wait(0.5)

        self.play(FadeOut(tree_lines), FadeOut(tree_pred), FadeOut(m3_title), FadeOut(prob3t), run_time=0.3)

        # --- Method 4: MLP/DNN Regression ---
        m3_title = Text("4. DNN Regression", font_size=20, color="#c0392b", weight=BOLD)
        m3_title.move_to(LEFT * 3.5 + UP * 2.5)
        self.play(Write(m3_title), run_time=0.3)

        # Neural network icon
        layers = VGroup()
        for lx, n_nodes in [(-5.0, 3), (-4.3, 4), (-3.6, 4), (-2.9, 2)]:
            for j in range(n_nodes):
                y_off = (j - (n_nodes - 1) / 2) * 0.35 - 0.2
                dot = Dot([lx, y_off, 0], radius=0.06, color="#7f8c8d", fill_opacity=0.5)
                layers.add(dot)
        self.play(FadeIn(layers), run_time=0.3)

        # Single output
        arrow_out = Arrow(LEFT * 2.6 + DOWN * 0.2, LEFT * 2.0 + DOWN * 0.2,
                         color="#c0392b", stroke_width=2, tip_length=0.1)
        pred_x = Cross(stroke_color="#c0392b", stroke_width=5).scale(0.12)
        pred_x.move_to(true_l + np.array([-0.2, 0.3, 0]))
        self.play(Create(arrow_out), Create(pred_x), run_time=0.4)

        # Error + no uncertainty
        err = DashedLine(pred_x.get_center(), true_l, color="#c0392b",
                        dash_length=0.05, stroke_width=1.5)
        prob3 = Text("Single point estimate\nNo uncertainty quantification", font_size=13, color="#c0392b")
        prob3.move_to(LEFT * 3.5 + DOWN * 2.5)
        self.play(Create(err), Write(prob3), run_time=0.4)

        # Big question mark
        q_mark = Text("?", font_size=50, color="#c0392b", weight=BOLD)
        q_mark.move_to(LEFT * 1.8 + DOWN * 0.8)
        self.play(Write(q_mark), run_time=0.3)
        self.wait(0.5)

        # ================================================================
        # RIGHT SIDE: Flow Matching (simultaneous with method 3 showing)
        # ================================================================

        # Particles from prior
        np.random.seed(42)
        n_p = 30
        starts = []
        for _ in range(n_p):
            while True:
                x = 3.5 + np.random.uniform(-1.8, 1.8)
                y = -0.2 + np.random.uniform(-1.2, 1.2)
                if ((x - 3.5) / 2.0) ** 2 + ((y + 0.2) / 1.4) ** 2 < 1:
                    starts.append(np.array([x, y, 0]))
                    break

        particles = [Dot(p, radius=0.055, color="#27ae60", fill_opacity=0.7) for p in starts]
        step_lbl = Text("t = 0  prior samples", font_size=16, color="#27ae60")
        step_lbl.move_to(RIGHT * 3.5 + UP * 2.5)

        self.play(Write(step_lbl), *[FadeIn(p, scale=0.3) for p in particles], run_time=0.6)
        self.wait(0.3)

        # Velocity field
        v_arrows = []
        for gx in np.linspace(1.8, 5.2, 6):
            for gy in np.linspace(-1.5, 1.1, 4):
                if ((gx - 3.5) / 2.0) ** 2 + ((gy + 0.2) / 1.4) ** 2 < 0.75:
                    s = np.array([gx, gy, 0])
                    d = true_r - s
                    d = d / (np.linalg.norm(d) + 0.01) * 0.3
                    arr = Arrow(s, s + d, color="#95a5a6", stroke_width=1, tip_length=0.06,
                               max_tip_length_to_length_ratio=0.3).set_opacity(0.25)
                    v_arrows.append(arr)
        v_grp = VGroup(*v_arrows)
        self.play(FadeIn(v_grp), run_time=0.4)

        # Flow particles
        ends = [true_r + np.array([np.random.randn() * 0.22, np.random.randn() * 0.18, 0])
                for _ in range(n_p)]

        for s in range(1, 5):
            t = s / 4
            interp = [starts[i] + t * (ends[i] - starts[i]) for i in range(n_p)]
            new_lbl = Text(f"t = {t:.2f}  ODE integration", font_size=16, color="#27ae60")
            new_lbl.move_to(RIGHT * 3.5 + UP * 2.5)
            self.play(
                *[p.animate.move_to(pos) for p, pos in zip(particles, interp)],
                ReplacementTransform(step_lbl, new_lbl),
                run_time=0.6, rate_func=smooth
            )
            step_lbl = new_lbl

        self.play(FadeOut(v_grp), run_time=0.2)

        # Posterior ellipse + labels
        unc = Ellipse(width=1.0, height=0.75, color="#27ae60", stroke_width=2.5, fill_opacity=0.06)
        unc.move_to(true_r)
        mean_dot = Dot(np.mean(ends, axis=0), radius=0.07, color="#2980b9")

        final_lbl = Text("Posterior p(x|y) + calibrated σ", font_size=16, color="#27ae60", weight=BOLD)
        final_lbl.move_to(RIGHT * 3.5 + UP * 2.5)

        self.play(
            Create(unc), FadeIn(mean_dot),
            ReplacementTransform(step_lbl, final_lbl),
            run_time=0.5
        )

        # σ arrows
        sx = DoubleArrow(true_r + LEFT * 0.45, true_r + RIGHT * 0.45,
                        color="#27ae60", stroke_width=1.5, tip_length=0.07)
        sy = DoubleArrow(true_r + DOWN * 0.33, true_r + UP * 0.33,
                        color="#27ae60", stroke_width=1.5, tip_length=0.07)
        self.play(Create(sx), Create(sy), run_time=0.3)

        # Bottom comparison labels
        left_summary = VGroup(
            Text("X  No uncertainty", font_size=15, color="#c0392b"),
            Text("X  Single point only", font_size=15, color="#c0392b"),
            Text("X  Geometry unaware", font_size=15, color="#c0392b"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(LEFT * 3.5 + DOWN * 3.2)

        right_summary = VGroup(
            Text("+  Full posterior distribution", font_size=15, color="#27ae60"),
            Text("+  Calibrated per-axis uncertainty", font_size=15, color="#27ae60"),
            Text("+  Geometry-constrained", font_size=15, color="#27ae60"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(RIGHT * 3.5 + DOWN * 3.2)

        self.play(
            FadeOut(prob3), FadeOut(q_mark),
            Write(left_summary), Write(right_summary),
            run_time=0.6
        )

        self.wait(2.5)
