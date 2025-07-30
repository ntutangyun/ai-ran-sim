#!/usr/bin/env python3
"""
Visualization Engine for AI RAN Simulation Evaluation
Loads evaluation results and prepares for chart generation.
"""

import os
import glob
import json
try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required for visualization_engine.py. Please install it with 'pip install pandas'.")
from typing import Dict, Any

class VisualizationEngine:
    def __init__(self, evaluation_results_dir: str):
        self.results_dir = evaluation_results_dir
        self.evaluation_data = None
        self.metrics_data = None
        self.load_evaluation_data()

    def plot_metric_radar(self):
        """Generate and save a radar chart of key evaluation metrics."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        self.calculate_enhanced_metrics()
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        academic_metrics = [
            'consistency_score', 'tool_selection_accuracy', 'context_retention_score',
            'knowledge_integration_score', 'reasoning_quality_score', 'adaptability_score',
            'confidence_calibration_score'
        ]
        avg_scores = []
        for metric in academic_metrics:
            if metric in self.metrics_data.columns:
                avg_scores.append(self.metrics_data[metric].mean())
            else:
                avg_scores.append(0.0)
        angles = np.linspace(0, 2 * np.pi, len(academic_metrics), endpoint=False).tolist()
        avg_scores += avg_scores[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, avg_scores, 'o-', linewidth=2, color='blue')
        ax.fill(angles, avg_scores, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in academic_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Metric Radar Chart', size=16, pad=20)
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'metric_radar.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved metric radar chart to {out_path}")

    def load_evaluation_data(self):
        """Load evaluation results from the latest JSON file in the directory."""
        json_files = glob.glob(os.path.join(self.results_dir, "dynamic_conversation_evaluation_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No evaluation results found in {self.results_dir}")
        latest_file = max(json_files, key=os.path.getctime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            self.evaluation_data = json.load(f)
        # Prepare a DataFrame for metrics (to be expanded in next steps)
        self.metrics_data = pd.DataFrame(self.evaluation_data.get('detailed_results', [])) 

    def summarize_data(self):
        """Print a summary of the loaded evaluation results."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        print(f"Loaded {len(self.metrics_data)} evaluation results.")
        print("Columns:", list(self.metrics_data.columns))
        print("Preview:")
        print(self.metrics_data.head()) 

    def plot_overall_performance(self):
        """Generate and save an overall performance indicator chart (average evaluation score)."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        import matplotlib.pyplot as plt
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        avg_score = self.metrics_data['evaluation_score'].mean()
        plt.figure(figsize=(6, 4))
        plt.bar(['Average Score'], [avg_score], color='skyblue')
        plt.ylim(0, 1)
        plt.ylabel('Evaluation Score')
        plt.title('Overall Agent Performance')
        plt.text(0, avg_score + 0.02, f"{avg_score:.2f}", ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'overall_performance.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved overall performance chart to {out_path}") 

    def plot_performance_by_difficulty(self):
        """Generate and save a bar chart of average evaluation score by difficulty."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        import matplotlib.pyplot as plt
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        diff_scores = self.metrics_data.groupby('difficulty')['evaluation_score'].mean()
        diff_scores_dict = diff_scores.to_dict()
        keys = [str(k) for k in diff_scores_dict.keys()]
        values = [float(v.item()) if hasattr(v, 'item') else float(v) for v in diff_scores_dict.values()]
        plt.figure(figsize=(6, 4))
        bars = plt.bar(keys, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.ylim(0, 1)
        plt.ylabel('Average Evaluation Score')
        plt.title('Performance by Difficulty')
        plt.xticks(rotation=0)
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'performance_by_difficulty.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved performance by difficulty chart to {out_path}") 

    def plot_performance_by_category(self):
        """Generate and save a bar chart of average evaluation score by category."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        import matplotlib.pyplot as plt
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        cat_scores = self.metrics_data.groupby('category')['evaluation_score'].mean()
        cat_scores_dict = cat_scores.to_dict()
        keys = [str(k) for k in cat_scores_dict.keys()]
        values = [float(v.item()) if hasattr(v, 'item') else float(v) for v in cat_scores_dict.values()]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(keys, values, color='teal')
        plt.ylim(0, 1)
        plt.ylabel('Average Evaluation Score')
        plt.title('Performance by Category')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'performance_by_category.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved performance by category chart to {out_path}") 

    def plot_tool_usage_analysis(self):
        """Generate and save a bar chart of tool usage counts."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        import matplotlib.pyplot as plt
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        # Flatten the list of tools used per question
        all_tools = []
        for tools in self.metrics_data['tools_used']:
            if isinstance(tools, list):
                all_tools.extend(tools)
        from collections import Counter
        tool_counts = Counter(all_tools)
        keys = list(tool_counts.keys())
        values = [tool_counts[k] for k in keys]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(keys, values, color='purple')
        plt.ylabel('Usage Count')
        plt.title('Tool Usage Analysis')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'tool_usage_analysis.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved tool usage analysis chart to {out_path}") 

    def plot_response_time_histogram(self):
        """Generate and save a histogram of response times."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        import matplotlib.pyplot as plt
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        response_times = self.metrics_data['response_time'].dropna().astype(float)
        plt.figure(figsize=(8, 4))
        plt.hist(response_times, bins=10, color='orange', edgecolor='black', alpha=0.7)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Count')
        plt.title('Response Time Distribution')
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'response_time_histogram.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved response time histogram to {out_path}") 

    def plot_error_handling_effectiveness(self):
        """Generate and save a bar chart of error handling effectiveness."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        import matplotlib.pyplot as plt
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        # Calculate error handling effectiveness (simplified metric)
        error_handling_scores = []
        for _, row in self.metrics_data.iterrows():
            question = str(row.get('static_question', '')).lower()
            response = str(row.get('agent_response', '')).lower()
            # Check if question contains error indicators
            error_indicators = ['error', 'not found', 'invalid', 'non-existent', 'missing']
            has_error_case = any(indicator in question for indicator in error_indicators)
            if has_error_case:
                # Check if response properly handles error
                error_handling_indicators = ['error', 'not found', 'invalid', 'does not exist', 'unavailable']
                handles_error = any(indicator in response for indicator in error_handling_indicators)
                error_handling_scores.append(1.0 if handles_error else 0.0)
            else:
                error_handling_scores.append(1.0)  # No error case to handle
        plt.figure(figsize=(6, 4))
        plt.hist(error_handling_scores, bins=[0, 0.5, 1.0], color='red', alpha=0.7, edgecolor='black')
        plt.xlabel('Error Handling Effectiveness')
        plt.ylabel('Count')
        plt.title('Error Handling Effectiveness Distribution')
        plt.xticks([0.25, 0.75], ['Poor', 'Good'])
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'error_handling_effectiveness.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved error handling effectiveness chart to {out_path}")

    def plot_correlation_analysis(self):
        """Generate and save a correlation heatmap of evaluation metrics."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        # Select numeric columns for correlation
        numeric_columns = self.metrics_data.select_dtypes(include=['number']).columns
        correlation_matrix = self.metrics_data[numeric_columns].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Metric Correlation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'correlation_analysis.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved correlation analysis to {out_path}")

    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all visualizations."""
        print("📊 Creating comprehensive evaluation dashboard...")
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import os
            vis_dir = os.path.join(self.results_dir, 'visualization')
            os.makedirs(vis_dir, exist_ok=True)
            # Create subplots for different sections
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Overall Performance', 'Performance by Difficulty',
                    'Performance by Category', 'Tool Usage Analysis',
                    'Response Time Analysis', 'Error Analysis'
                )
            )
            # Use a bar for overall score instead of indicator
            overall_score = self.metrics_data['evaluation_score'].mean()
            fig.add_trace(
                go.Bar(x=['Average Score'], y=[overall_score], marker_color='darkblue', name='Average Score'),
                row=1, col=1
            )
            # Performance by difficulty
            diff_scores = self.metrics_data.groupby('difficulty')['evaluation_score'].mean()
            fig.add_trace(
                go.Bar(x=list(diff_scores.index), y=list(diff_scores.values),
                       name="Difficulty Scores", marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']),
                row=1, col=2
            )
            # Performance by category
            cat_scores = self.metrics_data.groupby('category')['evaluation_score'].mean()
            fig.add_trace(
                go.Bar(x=list(cat_scores.index), y=list(cat_scores.values),
                       name="Category Scores", marker_color='#d62728'),
                row=2, col=1
            )
            # Tool usage
            all_tools = []
            for tools in self.metrics_data['tools_used']:
                if isinstance(tools, list):
                    all_tools.extend(tools)
            from collections import Counter
            tool_counts = Counter(all_tools)
            fig.add_trace(
                go.Bar(x=list(tool_counts.keys()), y=list(tool_counts.values()),
                       name="Tool Usage", marker_color='#9467bd'),
                row=2, col=2
            )
            # Response time histogram
            response_times = self.metrics_data['response_time'].dropna()
            fig.add_trace(
                go.Histogram(x=response_times, name="Response Time Distribution",
                            nbinsx=10, marker_color='#8c564b'),
                row=3, col=1
            )
            # Error handling
            error_scores = []
            for _, row in self.metrics_data.iterrows():
                question = str(row.get('static_question', '')).lower()
                response = str(row.get('agent_response', '')).lower()
                error_indicators = ['error', 'not found', 'invalid', 'non-existent', 'missing']
                has_error_case = any(indicator in question for indicator in error_indicators)
                if has_error_case:
                    error_handling_indicators = ['error', 'not found', 'invalid', 'does not exist', 'unavailable']
                    handles_error = any(indicator in response for indicator in error_handling_indicators)
                    error_scores.append(1.0 if handles_error else 0.0)
                else:
                    error_scores.append(1.0)
            error_counts = pd.Series(error_scores).value_counts()
            fig.add_trace(
                go.Bar(x=['Good', 'Poor'], y=[error_counts.get(1.0, 0), error_counts.get(0.0, 0)],
                       name="Error Handling", marker_color='#e377c2'),
                row=3, col=2
            )
            # Update layout
            fig.update_layout(height=1200, width=1000, title_text="Comprehensive AI Agent Evaluation Dashboard")
            # Save the dashboard
            dashboard_file = os.path.join(vis_dir, "comprehensive_dashboard.html")
            fig.write_html(dashboard_file)
            print(f"✅ Dashboard saved to: {dashboard_file}")
            return fig
        except Exception as e:
            print(f"Error creating comprehensive dashboard: {e}")
            return None

    def plot_network_metrics_heatmap(self):
        """Generate and save a heatmap of network-specific metrics."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        
        # Calculate enhanced metrics first
        self.calculate_enhanced_metrics()
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Network-specific metrics
        network_metrics = [
            'network_knowledge_accuracy', 'real_time_response_quality',
            'multi_layer_understanding_score', 'optimization_suggestion_quality'
        ]
        
        # Filter available metrics
        available_metrics = [m for m in network_metrics if m in self.metrics_data.columns]
        
        if not available_metrics:
            print("No network metrics available.")
            return
        
        # Create heatmap data
        heatmap_data = self.metrics_data[available_metrics].mean().to_frame().T
        
        plt.figure(figsize=(12, 4))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Network-Specific Metrics Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'network_metrics_heatmap.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved network metrics heatmap to {out_path}")

    def plot_semantic_analysis(self):
        """Generate and save semantic analysis visualization."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        self.calculate_enhanced_metrics()
        import matplotlib.pyplot as plt
        import os
        # Ensure visualization directory exists
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        semantic_metrics = [
            'semantic_similarity_score', 'factual_consistency_score',
            'logical_coherence_score', 'domain_expertise_score'
        ]
        available_metrics = [m for m in semantic_metrics if m in self.metrics_data.columns]
        if not available_metrics:
            print("No semantic metrics available.")
            return
        avg_scores = [self.metrics_data[m].mean() for m in available_metrics]
        metric_names = [m.replace('_', ' ').title() for m in available_metrics]
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        try:
        plt.figure(figsize=(10, 6))
            if len(metric_names) == 1:
                bars = plt.bar(metric_names, avg_scores, color=base_colors[0])
            else:
                color_list = (base_colors * ((len(metric_names) + len(base_colors) - 1) // len(base_colors)))[:len(metric_names)]
                bars = plt.bar(metric_names, avg_scores, color=color_list)
        plt.ylim(0, 1)
        plt.ylabel('Average Score')
        plt.title('Semantic Analysis Metrics')
        plt.xticks(rotation=45, ha='right')
        for bar, score in zip(bars, avg_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        plt.tight_layout()
            out_path = os.path.join(vis_dir, 'semantic_analysis.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved semantic analysis chart to {out_path}")
        except Exception as e:
            print(f"Error plotting semantic analysis: {e}")
            print(f"metric_names: {metric_names}")

    def plot_metric_comparison_matrix(self):
        """Generate and save a comparison matrix of all metrics."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        
        # Calculate enhanced metrics first
        self.calculate_enhanced_metrics()
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        
        # All metrics to compare
        all_metrics = [
            'evaluation_score', 'accuracy_score', 'completeness_score', 'relevance_score',
            'clarity_score', 'helpfulness_score', 'consistency_score', 'tool_selection_accuracy',
            'error_handling_effectiveness', 'context_retention_score', 'knowledge_integration_score',
            'reasoning_quality_score', 'adaptability_score', 'confidence_calibration_score',
            'network_knowledge_accuracy', 'real_time_response_quality', 'multi_layer_understanding_score',
            'optimization_suggestion_quality', 'semantic_similarity_score', 'factual_consistency_score',
            'logical_coherence_score', 'domain_expertise_score', 'adaptive_behavior_score'
        ]
        
        # Filter available metrics
        available_metrics = [m for m in all_metrics if m in self.metrics_data.columns]
        
        if len(available_metrics) < 2:
            print("Not enough metrics available for comparison.")
            return
        
        # Calculate correlation matrix
        correlation_matrix = self.metrics_data[available_metrics].corr()
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Metric Comparison Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'metric_comparison_matrix.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved metric comparison matrix to {out_path}")

    def plot_performance_trends(self):
        """Generate and save performance trend analysis."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        self.calculate_enhanced_metrics()
        import matplotlib.pyplot as plt
        import os
        vis_dir = os.path.join(self.results_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        # Extract question numbers for x-axis
        question_ids = self.metrics_data['question_id'].tolist()
        question_numbers = []
        for qid in question_ids:
            try:
                num = int(qid.split('_')[-1])
                question_numbers.append(num)
            except:
                question_numbers.append(len(question_numbers) + 1)
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # Overall score trend
        axes[0, 0].plot(question_numbers, self.metrics_data['evaluation_score'], 
                        marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Evaluation Score Trend')
        axes[0, 0].set_xlabel('Question Number')
        axes[0, 0].set_ylabel('Evaluation Score')
        axes[0, 0].grid(True, alpha=0.3)
        # Response time trend
        axes[0, 1].plot(question_numbers, self.metrics_data['response_time'], 
                        marker='s', linewidth=2, markersize=6, color='orange')
        axes[0, 1].set_title('Response Time Trend')
        axes[0, 1].set_xlabel('Question Number')
        axes[0, 1].set_ylabel('Response Time (seconds)')
        axes[0, 1].grid(True, alpha=0.3)
        # Tool usage trend
        tool_counts = [len(tools) if isinstance(tools, list) else 0 for tools in self.metrics_data['tools_used']]
        axes[1, 0].plot(question_numbers, tool_counts, 
                        marker='^', linewidth=2, markersize=6, color='green')
        axes[1, 0].set_title('Tool Usage Trend')
        axes[1, 0].set_xlabel('Question Number')
        axes[1, 0].set_ylabel('Number of Tools Used')
        axes[1, 0].grid(True, alpha=0.3)
        # Consistency trend (if available)
        if 'consistency_score' in self.metrics_data.columns:
            axes[1, 1].plot(question_numbers, self.metrics_data['consistency_score'], 
                            marker='d', linewidth=2, markersize=6, color='red')
            axes[1, 1].set_title('Consistency Score Trend')
            axes[1, 1].set_xlabel('Question Number')
            axes[1, 1].set_ylabel('Consistency Score')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Alternative: domain expertise trend
            if 'domain_expertise_score' in self.metrics_data.columns:
                axes[1, 1].plot(question_numbers, self.metrics_data['domain_expertise_score'], 
                                marker='d', linewidth=2, markersize=6, color='purple')
                axes[1, 1].set_title('Domain Expertise Trend')
                axes[1, 1].set_xlabel('Question Number')
                axes[1, 1].set_ylabel('Domain Expertise Score')
                axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(vis_dir, 'performance_trends.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved performance trends to {out_path}")

    def generate_enhanced_dashboard(self):
        """Generate enhanced dashboard with all academic metrics."""
        print("🎓 Generating enhanced academic dashboard...")
        
        # Calculate enhanced metrics first
        self.calculate_enhanced_metrics()
        
        # Generate all enhanced visualizations
        self.plot_metric_radar()
        self.plot_network_metrics_heatmap()
        self.plot_semantic_analysis()
        self.plot_metric_comparison_matrix()
        self.plot_performance_trends()
        
        print("✅ Enhanced academic dashboard generated successfully!")

    def generate_all_charts(self):
        """Generate all available charts including enhanced academic metrics."""
        print("🎨 Generating all evaluation charts...")
        
        # Basic charts
        self.plot_overall_performance()
        self.plot_performance_by_difficulty()
        self.plot_performance_by_category()
        self.plot_tool_usage_analysis()
        self.plot_response_time_histogram()
        self.plot_error_handling_effectiveness()
        self.plot_correlation_analysis()
        
        # Enhanced academic charts
        self.generate_enhanced_dashboard()
        
        # Interactive dashboard
        self.create_comprehensive_dashboard()
        
        print("✅ All charts generated successfully!")

    def calculate_enhanced_metrics(self):
        """Calculate enhanced academic metrics for each evaluation result."""
        if self.metrics_data is None or self.metrics_data.empty:
            print("No evaluation data loaded.")
            return
        
        enhanced_metrics = []
        
        for _, row in self.metrics_data.iterrows():
            metrics = {}
            
            # Basic metrics (already available)
            metrics['accuracy_score'] = row.get('evaluation_score', 0.0)
            metrics['completeness_score'] = self._calculate_completeness_score(row)
            metrics['relevance_score'] = self._calculate_relevance_score(row)
            metrics['clarity_score'] = self._calculate_clarity_score(row)
            metrics['helpfulness_score'] = self._calculate_helpfulness_score(row)
            
            # Advanced academic metrics
            metrics['consistency_score'] = self._calculate_consistency_score(row)
            metrics['tool_selection_accuracy'] = self._calculate_tool_selection_accuracy(row)
            metrics['error_handling_effectiveness'] = self._calculate_error_handling_effectiveness(row)
            metrics['context_retention_score'] = self._calculate_context_retention_score(row)
            metrics['knowledge_integration_score'] = self._calculate_knowledge_integration_score(row)
            metrics['reasoning_quality_score'] = self._calculate_reasoning_quality_score(row)
            metrics['adaptability_score'] = self._calculate_adaptability_score(row)
            metrics['confidence_calibration_score'] = self._calculate_confidence_calibration_score(row)
            
            # Network-specific metrics
            metrics['network_knowledge_accuracy'] = self._calculate_network_knowledge_accuracy(row)
            metrics['real_time_response_quality'] = self._calculate_real_time_response_quality(row)
            metrics['multi_layer_understanding_score'] = self._calculate_multi_layer_understanding_score(row)
            metrics['optimization_suggestion_quality'] = self._calculate_optimization_suggestion_quality(row)
            
            # Semantic analysis metrics
            metrics['semantic_similarity_score'] = self._calculate_semantic_similarity_score(row)
            metrics['factual_consistency_score'] = self._calculate_factual_consistency_score(row)
            metrics['logical_coherence_score'] = self._calculate_logical_coherence_score(row)
            metrics['domain_expertise_score'] = self._calculate_domain_expertise_score(row)
            metrics['adaptive_behavior_score'] = self._calculate_adaptive_behavior_score(row)
            
            enhanced_metrics.append(metrics)
        
        # Add enhanced metrics to the DataFrame
        enhanced_df = pd.DataFrame(enhanced_metrics)
        self.metrics_data = pd.concat([self.metrics_data, enhanced_df], axis=1)
        print(f"✅ Enhanced metrics calculated for {len(self.metrics_data)} evaluation results")

    def _calculate_completeness_score(self, row):
        """Calculate completeness score based on response length and content."""
        response = str(row.get('agent_response', ''))
        question = str(row.get('static_question', ''))
        
        # Simple heuristic: response should be proportional to question complexity
        question_words = len(question.split())
        response_words = len(response.split())
        
        if question_words == 0:
            return 0.5
        
        expected_ratio = 3.0  # Expected response to question word ratio
        actual_ratio = response_words / question_words
        
        if actual_ratio >= expected_ratio * 0.8:
            return 1.0
        elif actual_ratio >= expected_ratio * 0.5:
            return 0.7
        else:
            return 0.3

    def _calculate_relevance_score(self, row):
        """Calculate relevance score based on keyword matching."""
        response = str(row.get('agent_response', '')).lower()
        question = str(row.get('static_question', '')).lower()
        
        # Extract key terms from question
        question_terms = set(question.split())
        response_terms = set(response.split())
        
        if not question_terms:
            return 0.5
        
        # Calculate overlap
        overlap = len(question_terms & response_terms)
        relevance = overlap / len(question_terms)
        
        return min(relevance * 2, 1.0)  # Scale up relevance

    def _calculate_clarity_score(self, row):
        """Calculate clarity score based on response structure."""
        response = str(row.get('agent_response', ''))
        
        # Check for clarity indicators
        clarity_indicators = [
            'clearly', 'specifically', 'in detail', 'as follows',
            'first', 'second', 'finally', 'therefore', 'thus'
        ]
        
        clarity_count = sum(1 for indicator in clarity_indicators if indicator in response.lower())
        
        if clarity_count >= 2:
            return 1.0
        elif clarity_count >= 1:
            return 0.8
        else:
            return 0.5

    def _calculate_helpfulness_score(self, row):
        """Calculate helpfulness score based on actionable content."""
        response = str(row.get('agent_response', '')).lower()
        
        # Check for helpful indicators
        helpful_indicators = [
            'you can', 'to do this', 'here is how', 'steps to',
            'recommend', 'suggest', 'should', 'need to', 'action'
        ]
        
        helpful_count = sum(1 for indicator in helpful_indicators if indicator in response)
        
        if helpful_count >= 2:
            return 1.0
        elif helpful_count >= 1:
            return 0.7
        else:
            return 0.4

    def _calculate_consistency_score(self, row):
        """Calculate consistency score based on response coherence."""
        response = str(row.get('agent_response', ''))
        
        # Check for consistency indicators
        consistency_indicators = [
            'consistent', 'matches', 'agrees with', 'corresponds',
            'in line with', 'according to', 'based on'
        ]
        
        consistency_count = sum(1 for indicator in consistency_indicators if indicator in response.lower())
        
        return min(consistency_count * 0.3 + 0.5, 1.0)

    def _calculate_tool_selection_accuracy(self, row):
        """Calculate tool selection accuracy."""
        expected_tools = row.get('expected_tools', [])
        actual_tools = row.get('tools_used', [])
        
        if not expected_tools:
            return 1.0
        
        if not isinstance(expected_tools, list):
            expected_tools = [expected_tools]
        if not isinstance(actual_tools, list):
            actual_tools = [actual_tools]
        
        correct_tools = set(expected_tools) & set(actual_tools)
        precision = len(correct_tools) / len(actual_tools) if actual_tools else 0.0
        recall = len(correct_tools) / len(expected_tools) if expected_tools else 0.0
        
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

    def _calculate_error_handling_effectiveness(self, row):
        """Calculate error handling effectiveness."""
        question = str(row.get('static_question', '')).lower()
        response = str(row.get('agent_response', '')).lower()
        
        error_indicators = ['error', 'not found', 'invalid', 'non-existent', 'missing']
        has_error_case = any(indicator in question for indicator in error_indicators)
        
        if has_error_case:
            error_handling_indicators = ['error', 'not found', 'invalid', 'does not exist', 'unavailable']
            handles_error = any(indicator in response for indicator in error_handling_indicators)
            return 1.0 if handles_error else 0.0
        
        return 1.0

    def _calculate_context_retention_score(self, row):
        """Calculate context retention score."""
        conversation = row.get('dynamic_conversation', [])
        if not isinstance(conversation, list) or len(conversation) < 2:
            return 1.0
        
        response_length = len(str(row.get('agent_response', '')))
        conversation_length = len(conversation)
        
        expected_length = 100 + (conversation_length * 50)
        actual_length = response_length
        
        if actual_length >= expected_length * 0.8:
            return 1.0
        elif actual_length >= expected_length * 0.5:
            return 0.7
        else:
            return 0.3

    def _calculate_knowledge_integration_score(self, row):
        """Calculate knowledge integration score."""
        tool_outputs = row.get('tool_outputs', {})
        response = str(row.get('agent_response', ''))
        
        if len(tool_outputs) > 1:
            integration_indicators = ['additionally', 'furthermore', 'moreover', 'also', 'while', 'however']
            has_integration = any(indicator in response.lower() for indicator in integration_indicators)
            return 1.0 if has_integration else 0.5
        elif len(tool_outputs) == 1:
            return 0.8
        else:
            return 0.3

    def _calculate_reasoning_quality_score(self, row):
        """Calculate reasoning quality score."""
        response = str(row.get('agent_response', '')).lower()
        
        reasoning_indicators = [
            'because', 'since', 'therefore', 'thus', 'as a result',
            'this means', 'which indicates', 'suggesting that',
            'analysis shows', 'based on', 'considering'
        ]
        
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response)
        
        if reasoning_count >= 3:
            return 1.0
        elif reasoning_count >= 2:
            return 0.8
        elif reasoning_count >= 1:
            return 0.6
        else:
            return 0.3

    def _calculate_adaptability_score(self, row):
        """Calculate adaptability score."""
        category = row.get('category', '')
        difficulty = row.get('difficulty', '')
        
        category_score = 0.8 if category in ['performance_analysis', 'network_analysis'] else 0.6
        difficulty_score = {'easy': 0.6, 'medium': 0.8, 'hard': 1.0}.get(difficulty, 0.5)
        
        return (category_score + difficulty_score) / 2

    def _calculate_confidence_calibration_score(self, row):
        """Calculate confidence calibration score."""
        response = str(row.get('agent_response', '')).lower()
        
        high_confidence_indicators = ['certainly', 'definitely', 'clearly', 'obviously']
        moderate_confidence_indicators = ['likely', 'probably', 'seems', 'appears']
        low_confidence_indicators = ['maybe', 'possibly', 'uncertain', 'unclear']
        
        high_count = sum(1 for indicator in high_confidence_indicators if indicator in response)
        moderate_count = sum(1 for indicator in moderate_confidence_indicators if indicator in response)
        low_count = sum(1 for indicator in low_confidence_indicators if indicator in response)
        
        total_indicators = high_count + moderate_count + low_count
        if total_indicators == 0:
            return 0.5
        
        confidence_level = (high_count * 1.0 + moderate_count * 0.6 + low_count * 0.2) / total_indicators
        return min(confidence_level, 1.0)

    def _calculate_network_knowledge_accuracy(self, row):
        """Calculate network knowledge accuracy."""
        response = str(row.get('agent_response', '')).lower()
        category = row.get('category', '')
        
        network_keywords = {
            'ue_status': ['ue', 'user equipment', 'connection', 'cell', 'signal'],
            'network_overview': ['base station', 'network', 'coverage', 'cell'],
            'ai_services': ['ai', 'service', 'intelligence', 'automation'],
            'performance_analysis': ['performance', 'throughput', 'latency', 'quality'],
            'ric_operations': ['ric', 'controller', 'orchestration'],
            'xapps': ['xapp', 'application', 'service']
        }
        
        expected_keywords = network_keywords.get(category, [])
        if not expected_keywords:
            return 0.8
        
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in response)
        return min(keyword_matches / len(expected_keywords), 1.0)

    def _calculate_real_time_response_quality(self, row):
        """Calculate real-time response quality."""
        response_time = row.get('response_time', 0.0)
        
        if response_time < 1.0:
            return 0.7
        elif response_time < 5.0:
            return 1.0
        elif response_time < 10.0:
            return 0.8
        else:
            return 0.5

    def _calculate_multi_layer_understanding_score(self, row):
        """Calculate multi-layer understanding score."""
        response = str(row.get('agent_response', '')).lower()
        
        layer_indicators = [
            'physical layer', 'data link', 'network layer', 'transport',
            'application layer', 'protocol', 'stack', 'layer'
        ]
        
        layer_mentions = sum(1 for indicator in layer_indicators if indicator in response)
        
        if layer_mentions >= 2:
            return 1.0
        elif layer_mentions >= 1:
            return 0.7
        else:
            return 0.4

    def _calculate_optimization_suggestion_quality(self, row):
        """Calculate optimization suggestion quality."""
        response = str(row.get('agent_response', '')).lower()
        category = row.get('category', '')
        
        if 'optimization' in category or 'analysis' in category:
            optimization_indicators = [
                'optimize', 'improve', 'enhance', 'recommend', 'suggest',
                'better', 'efficient', 'performance', 'upgrade'
            ]
            
            optimization_count = sum(1 for indicator in optimization_indicators if indicator in response)
            
            if optimization_count >= 2:
                return 1.0
            elif optimization_count >= 1:
                return 0.7
            else:
                return 0.3
        else:
            return 0.8

    def _calculate_semantic_similarity_score(self, row):
        """Calculate semantic similarity score."""
        question = str(row.get('static_question', '')).lower()
        response = str(row.get('agent_response', '')).lower()
        
        question_words = set(question.split())
        response_words = set(response.split())
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words & response_words)
        similarity = overlap / len(question_words)
        
        return min(similarity, 1.0)

    def _calculate_factual_consistency_score(self, row):
        """Calculate factual consistency score."""
        tool_outputs = row.get('tool_outputs', {})
        response = str(row.get('agent_response', ''))
        
        if not tool_outputs:
            return 0.7
        
        consistency_score = 0.0
        for tool_name, output in tool_outputs.items():
            if isinstance(output, dict):
                output_str = str(output).lower()
            else:
                output_str = str(output).lower()
            
            response_lower = response.lower()
            if any(key in response_lower for key in output_str.split()[:5]):
                consistency_score += 1.0
        
        return min(consistency_score / len(tool_outputs), 1.0) if tool_outputs else 0.7

    def _calculate_logical_coherence_score(self, row):
        """Calculate logical coherence score."""
        response = str(row.get('agent_response', '')).lower()
        
        logical_connectors = [
            'because', 'therefore', 'thus', 'as a result', 'consequently',
            'however', 'nevertheless', 'on the other hand', 'meanwhile'
        ]
        
        connector_count = sum(1 for connector in logical_connectors if connector in response)
        
        if connector_count >= 2:
            return 1.0
        elif connector_count >= 1:
            return 0.8
        else:
            return 0.5

    def _calculate_domain_expertise_score(self, row):
        """Calculate domain expertise score."""
        response = str(row.get('agent_response', '')).lower()
        
        domain_terms = [
            'sinr', 'cqi', 'mcs', 'qos', '5qi', 'urllc', 'embb', 'mmtc',
            'handover', 'roaming', 'latency', 'throughput', 'bandwidth',
            'frequency', 'spectrum', 'antenna', 'beamforming', 'mimo'
        ]
        
        domain_term_count = sum(1 for term in domain_terms if term in response)
        
        if domain_term_count >= 3:
            return 1.0
        elif domain_term_count >= 2:
            return 0.8
        elif domain_term_count >= 1:
            return 0.6
        else:
            return 0.4

    def _calculate_adaptive_behavior_score(self, row):
        """Calculate adaptive behavior score."""
        difficulty = row.get('difficulty', '')
        tools_used = row.get('tools_used', [])
        
        difficulty_score = {'easy': 0.6, 'medium': 0.8, 'hard': 1.0}.get(difficulty, 0.5)
        tool_score = min(len(tools_used) / 2, 1.0) if isinstance(tools_used, list) else 0.5
        
        return (difficulty_score + tool_score) / 2

# Remove CLI, always run all charts
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python visualization_engine.py <results_dir>")
        sys.exit(1)
    results_dir = sys.argv[1]
    viz_engine = VisualizationEngine(results_dir)
    viz_engine.generate_all_charts()
    viz_engine.plot_metric_radar()
    print("✅ All charts generated successfully!") 