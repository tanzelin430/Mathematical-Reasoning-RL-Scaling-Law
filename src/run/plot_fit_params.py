import matplotlib.pyplot as plt
import numpy as np
from src.common import config
from src.common import plot

def plot_fit_params(args, fitters):
    """Entry point for plotting fit parameters"""
    if not args.fit_param_plot_schema:
        return
    PLOT_PARAMS_SCHEMA[args.fit_param_plot_schema](args, fitters)

def plot_fit_params_single(args, fitters):
    """Single schema: one plot per param per source per fit_x"""
    for fitter in fitters:
        context = fitter.get_context()
        data_source, curve_name, fit_x = context["data_source"], context['fit_curve'], context['fit_x']
        lookup_params = fitter.get_lookup_params()
        if lookup_params is not None: 
            curve_label = config.DEFAULT_LABELS[curve_name]
            for param_name, param_table in lookup_params.items():
                x = np.array(sorted(param_table.keys()))
                y = np.array([param_table[k] for k in x])
                print(f"\n{param_name}({curve_name}):  ")
                print(f"  x: {x}")
                print(f"  y: {y}")

                # Plot param(curve_column)
                fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=300)
                ax1 = plot.plot_basic(
                    x=x,
                    y=y,
                    use_scatter=True,
                    scatter_s=150,
                    color=config.get_color_for_curve(param_name),
                    ax=ax1
                )
                ax1 = plot.plot_basic_settings(
                    ax=ax1,
                    x_scale="log",
                    x_label=f"{curve_label}",
                    y_label=f"{param_name}({curve_name})",
                    title=f"{param_name}({curve_name}) vs {curve_label}",
                    use_legend=False,
                    x_margin=0.1, y_margin=0.1,
                    x_tick_spacing=args.x_tick_spacing,
                    y_tick_spacing=args.y_tick_spacing,
                    x_grid_spacing=args.x_grid_spacing,
                    y_grid_spacing=args.y_grid_spacing
                )
                
                plt.tight_layout(pad=1.0)
                eval_file_str = config.TEST_EVALS[args.eval]['file_str']
                
                # Ensure output directory exists
                args.output_base_dir.mkdir(parents=True, exist_ok=True)
                
                param_plot_path = args.output_base_dir / f"{args.output_prefix}{data_source}_{eval_file_str}_{curve_name}_{fit_x}_{param_name}_scatter.pdf"
                plt.savefig(param_plot_path, bbox_inches='tight', dpi=300)
                print(f"Saved {param_name}({curve_name}) plot: {param_plot_path}")
                plt.close()


def plot_fit_params_compare_source(args, fitters):
    """Compare-source schema: overlay all data sources in one plot
    
    Creates one file with N subplots (one per param), comparing all data_sources.
    If multiple fit_x, creates separate files for each fit_x.
    """
    # Group fitters by (fit_curve, fit_x)
    fitter_groups = {}
    for fitter in fitters:
        context = fitter.get_context()
        key = (context['fit_curve'], context['fit_x'])
        if key not in fitter_groups:
            fitter_groups[key] = []
        fitter_groups[key].append(fitter)
    
    # Process each group
    for (curve_name, fit_x), group_fitters in fitter_groups.items():
        print(f"\n--- Compare-source: {curve_name} vs {fit_x} ---")
        
        # Collect all param names (with custom sorting)
        all_param_names = set()
        for fitter in group_fitters:
            lookup_params = fitter.get_lookup_params()
            if lookup_params:
                all_param_names.update(lookup_params.keys())
        
        if not all_param_names:
            print(f"No parameters found for {curve_name} vs {fit_x}")
            continue
        
        params_to_plot = _sort_param_names(all_param_names)  # Use custom sort
        n_params = len(params_to_plot)
        
        # Create figure with subplots for all params
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 4), dpi=300)
        if n_params == 1:
            axes = [axes]
        
        curve_label = config.DEFAULT_LABELS.get(curve_name, curve_name)
        curve_short = config.DEFAULT_SHORT_NAME.get(curve_name, curve_name)
        fit_x_short = config.DEFAULT_SHORT_NAME.get(fit_x, fit_x)
        
        # Plot each parameter
        for ax_idx, param_name in enumerate(params_to_plot):
            ax = axes[ax_idx]
            
            # Plot each data source
            for fitter in group_fitters:
                context = fitter.get_context()
                data_source = context['data_source']
                lookup_params = fitter.get_lookup_params()
                
                if not lookup_params or param_name not in lookup_params:
                    continue
                
                param_table = lookup_params[param_name]
                x = np.array(sorted(param_table.keys()))
                y = np.array([param_table[k] for k in x])
                
                # Plot
                plot.plot_basic(
                    x=x,
                    y=y,
                    use_scatter=True,
                    scatter_alpha=args.scatter_alpha,
                    scatter_s=args.scatter_size,
                    scatter_marker=config.DEFAULT_MARKERS.get(data_source, 'o'),
                    color=config.COLOR_MAPPING.get(data_source, 'blue'),
                    ax=ax
                )
                
                # Add legend entry
                ax.scatter([], [], 
                          color=config.COLOR_MAPPING.get(data_source, 'blue'),
                          marker=config.DEFAULT_MARKERS.get(data_source, 'o'),
                          s=args.scatter_size,
                          label=config.DEFAULT_LABELS.get(data_source, data_source))
            
            # Apply settings with args parameters
            plot.plot_basic_settings(
                ax=ax,
                x_scale=args.plot_x_scale,
                y_scale=args.plot_y_scale,
                x_label=f'{curve_label} ({curve_short})',
                y_label=f'${param_name}_{{{fit_x_short}}}({curve_short})$',
                use_legend=args.plot_use_legend,
                legend_loc=args.plot_legend_loc,
                legend_bbox_to_anchor=args.plot_legend_bbox_to_anchor,
                x_tick_on_data=True,
                x_tick_spacing=args.x_tick_spacing,
                y_tick_spacing=args.y_tick_spacing,
                x_grid_spacing=args.x_grid_spacing,
                y_grid_spacing=args.y_grid_spacing,
            )
        
        plt.tight_layout()
        
        # Save
        first_eval = group_fitters[0].get_context()['eval'] # should be the same for all fitters, if multiple, use the first

        args.output_base_dir.mkdir(parents=True, exist_ok=True)
        eval_file_str = config.TEST_EVALS[first_eval]['file_str']
        sources = [f.get_context()['data_source'] for f in group_fitters]
        sources_str = "_vs_".join(sources)
        params_str = "_".join(params_to_plot)
        
        save_path = args.output_base_dir / (
            f"{args.output_prefix}{sources_str}_{eval_file_str}_"
            f"{curve_name}_{fit_x}_{params_str}_compare_source.pdf"
        )
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        plt.close()


def plot_fit_params_compare_x(args, fitters):
    """Compare-x schema: overlay all L(curve_column, x) in one plot
    
    Creates one file per data_source with N subplots (one per param), comparing different fit_x.
    If multiple data_sources, creates separate files for each.
    """
    # Group fitters by (data_source, fit_curve)
    fitter_groups = {}
    for fitter in fitters:
        context = fitter.get_context()
        key = (context['data_source'], context['fit_curve'])
        if key not in fitter_groups:
            fitter_groups[key] = []
        fitter_groups[key].append(fitter)
    
    # Process each group
    for (data_source, curve_name), group_fitters in fitter_groups.items():
        print(f"\n--- Compare-x: {data_source}, {curve_name} ---")
        
        # Check if we have multiple fit_x to compare
        fit_x_list = [f.get_context()['fit_x'] for f in group_fitters]
        if len(set(fit_x_list)) < 2:
            print(f"Warning: Only {len(set(fit_x_list))} unique fit_x found, "
                  f"compare-x is most useful with multiple fit_x columns")
        
        # Collect all param names (with custom sorting)
        all_param_names = set()
        for fitter in group_fitters:
            lookup_params = fitter.get_lookup_params()
            if lookup_params:
                all_param_names.update(lookup_params.keys())
        
        if not all_param_names:
            print(f"No parameters found for {data_source}, {curve_name}")
            continue
        
        params_to_plot = _sort_param_names(all_param_names)  # Use custom sort
        n_params = len(params_to_plot)
        
        # Create figure with subplots for all params
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 4), dpi=300)
        if n_params == 1:
            axes = [axes]
        
        curve_label = config.DEFAULT_LABELS.get(curve_name, curve_name)
        curve_short = config.DEFAULT_SHORT_NAME.get(curve_name, curve_name)
        
        # Plot each parameter
        for ax_idx, param_name in enumerate(params_to_plot):
            ax = axes[ax_idx]
            
            # Plot each fit_x
            for fitter in group_fitters:
                context = fitter.get_context()
                fit_x = context['fit_x']
                lookup_params = fitter.get_lookup_params()
                
                if not lookup_params or param_name not in lookup_params:
                    continue
                
                param_table = lookup_params[param_name]
                x = np.array(sorted(param_table.keys()))
                y = np.array([param_table[k] for k in x])
                
                # Use different colors/markers for different fit_x
                # Try to get from config, otherwise use default
                color = config.get_color_for_curve(fit_x)
                marker = config.DEFAULT_MARKERS.get(fit_x, 'o')
                
                # Plot
                plot.plot_basic(
                    x=x,
                    y=y,
                    use_scatter=True,
                    scatter_alpha=args.scatter_alpha,
                    scatter_s=args.scatter_size,
                    scatter_marker=marker,
                    color=color,
                    ax=ax
                )
                
                # Add legend entry with fit_x info
                fit_x_short = config.DEFAULT_SHORT_NAME.get(fit_x, fit_x)
                ax.scatter([], [], 
                          color=color,
                          marker=marker,
                          s=args.scatter_size,
                          label=f'L({curve_name},{fit_x_short})')
            
            # Apply settings with args parameters
            plot.plot_basic_settings(
                ax=ax,
                x_scale=args.plot_x_scale,
                y_scale=args.plot_y_scale,
                x_label=f'{curve_label} ({curve_short})',
                y_label=f'${param_name}({curve_short})$',
                use_legend=args.plot_use_legend,
                legend_loc=args.plot_legend_loc,
                x_tick_on_data=True,
                x_tick_spacing=args.x_tick_spacing,
                y_tick_spacing=args.y_tick_spacing,
                x_grid_spacing=args.x_grid_spacing,
                y_grid_spacing=args.y_grid_spacing,
            )
        
        plt.tight_layout()
        
        # Save
        args.output_base_dir.mkdir(parents=True, exist_ok=True)
        first_eval = group_fitters[0].get_context()['eval'] # should be the same for all fitters, if multiple, use the first
        eval_file_str = config.TEST_EVALS[first_eval]['file_str']
        fit_x_list_str = "_vs_".join(sorted(set(fit_x_list)))
        params_str = "_".join(params_to_plot)
        
        save_path = args.output_base_dir / (
            f"{args.output_prefix}{data_source}_{eval_file_str}_"
            f"{curve_name}_{fit_x_list_str}_{params_str}_compare_x.pdf"
        )
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        plt.close()


def _sort_param_names(param_names):
    """Sort parameter names with custom order: k before E0, then alphabetically"""
    def sort_key(name):
        if name.startswith('k'):
            return (0, name)
        elif name.startswith('E'):
            return (1, name)
        else:
            return (2, name)
    
    return sorted(param_names, key=sort_key)


def plot_fit_params_table(args, fitters):
    """Table schema: Generate LaTeX table for fitted parameters (regular version)"""
    
    print("\n" + "="*80)
    print("LaTeX Tables: Fitting Parameters (Regular Version)")
    print("="*80)
    
    # Group fitters by (fit_curve, fit_x)
    fitter_groups = {}
    for fitter in fitters:
        context = fitter.get_context()
        key = (context['fit_curve'], context['fit_x'])
        if key not in fitter_groups:
            fitter_groups[key] = []
        fitter_groups[key].append(fitter)
    
    # Process each group (each fit_x gets its own table)
    for (curve_name, fit_x), group_fitters in fitter_groups.items():
        # Collect all param names, curve values, and build data structure
        all_param_names = set()
        all_curve_vals = set()
        
        # Build complete data map: {data_source: {curve_val: {param: val, 'r2': val}}}
        data_map = {}
        for fitter in group_fitters:
            context = fitter.get_context()
            data_source = context['data_source']
            info = fitter.get_info()
            r2 = info.get('r2', None)
            
            lookup_params = fitter.get_lookup_params()
            if not lookup_params:
                continue
            
            all_param_names.update(lookup_params.keys())
            
            data_map[data_source] = {}
            # Get all curve values from first param
            first_param_table = list(lookup_params.values())[0]
            all_curve_vals.update(first_param_table.keys())
            
            for curve_val in first_param_table.keys():
                data_map[data_source][curve_val] = {'r2': r2}
                for param_name, param_table in lookup_params.items():
                    if curve_val in param_table:
                        data_map[data_source][curve_val][param_name] = param_table[curve_val]
        
        if not all_param_names:
            continue
        
        all_curve_vals = sorted(all_curve_vals)
        all_param_names = _sort_param_names(all_param_names)  # Custom sort
        data_sources = sorted([f.get_context()['data_source'] for f in group_fitters])
        
        # Get short names for display
        curve_short = config.DEFAULT_SHORT_NAME.get(curve_name, curve_name)
        fit_x_short = config.DEFAULT_SHORT_NAME.get(fit_x, fit_x)
        
        # Generate Regular version (all models in rows)
        _print_table_regular(curve_name, fit_x, curve_short, fit_x_short, 
                           all_curve_vals, all_param_names, data_sources, data_map)
    
    print("\n" + "="*80)
    print("Note: Include \\usepackage{booktabs} and \\usepackage{float} in LaTeX preamble")
    print("="*80)


def plot_fit_params_table_compact(args, fitters):
    """Table-compact schema: Generate LaTeX table for fitted parameters (compact version)"""
    
    print("\n" + "="*80)
    print("LaTeX Tables: Fitting Parameters (Compact Version)")
    print("="*80)
    
    # Group fitters by (fit_curve, fit_x)
    fitter_groups = {}
    for fitter in fitters:
        context = fitter.get_context()
        key = (context['fit_curve'], context['fit_x'])
        if key not in fitter_groups:
            fitter_groups[key] = []
        fitter_groups[key].append(fitter)
    
    # Process each group (each fit_x gets its own table)
    for (curve_name, fit_x), group_fitters in fitter_groups.items():
        # Collect all param names, curve values, and build data structure
        all_param_names = set()
        all_curve_vals = set()
        
        # Build complete data map: {data_source: {curve_val: {param: val, 'r2': val}}}
        data_map = {}
        for fitter in group_fitters:
            context = fitter.get_context()
            data_source = context['data_source']
            info = fitter.get_info()
            r2 = info.get('r2', None)
            
            lookup_params = fitter.get_lookup_params()
            if not lookup_params:
                continue
            
            all_param_names.update(lookup_params.keys())
            
            data_map[data_source] = {}
            # Get all curve values from first param
            first_param_table = list(lookup_params.values())[0]
            all_curve_vals.update(first_param_table.keys())
            
            for curve_val in first_param_table.keys():
                data_map[data_source][curve_val] = {'r2': r2}
                for param_name, param_table in lookup_params.items():
                    if curve_val in param_table:
                        data_map[data_source][curve_val][param_name] = param_table[curve_val]
        
        if not all_param_names:
            continue
        
        all_curve_vals = sorted(all_curve_vals)
        all_param_names = _sort_param_names(all_param_names)  # Custom sort
        data_sources = sorted([f.get_context()['data_source'] for f in group_fitters])
        
        # Get short names for display
        curve_short = config.DEFAULT_SHORT_NAME.get(curve_name, curve_name)
        fit_x_short = config.DEFAULT_SHORT_NAME.get(fit_x, fit_x)
        
        # Generate Compact version (model sizes in rows, sources in columns)
        _print_table_compact(curve_name, fit_x, curve_short, fit_x_short,
                           all_curve_vals, all_param_names, data_sources, data_map)
    
    print("\n" + "="*80)
    print("Note: Include \\usepackage{booktabs}, \\usepackage{multirow}, and \\usepackage{float} in LaTeX preamble")
    print("="*80)


def _print_table_regular(curve_name, fit_x, curve_short, fit_x_short, 
                         all_curve_vals, all_param_names, data_sources, data_map):
    """Print regular table version: each model as a row"""
    
    print(f"\n\\begin{{table}}[H]")
    print("\\centering")
    print(f"\\caption{{$L({curve_short},{fit_x_short})$ Fitting Results}}")
    print(f"\\label{{tab:fitting_results_{curve_short}_{fit_x_short}}}")
    
    # Build column spec: Model + params + R²
    n_cols = 1 + len(all_param_names) + 1  # Model + params + R²
    col_spec = "l" + "l" * len(all_param_names) + "l"
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    
    # Header
    header_parts = ["\\textbf{Model}"]
    for param_name in all_param_names:
        header_parts.append(f"\\textbf{{${param_name}_{{{fit_x_short}}}$}}")
    header_parts.append(f"\\textbf{{$R^2_{{{fit_x_short}}}$}}")
    print(" & ".join(header_parts) + " \\\\")
    print("\\midrule")
    
    # Rows: iterate through data_sources first, then curve_vals within each
    for data_source in data_sources:
        # Use capitalized short form without "Model" suffix
        ds_short = data_source.capitalize() if data_source in ['base', 'instruct'] else data_source
        for curve_val in all_curve_vals:
            # Format curve value
            if curve_val >= 1e6:
                curve_str = (f"{curve_val/1e9:.1f}B" if curve_val/1e9 != int(curve_val/1e9) 
                           else f"{int(curve_val/1e9)}B")
            else:
                curve_str = f"{curve_val:.0f}"
            
            row_parts = [f"{curve_str}-{ds_short}"]
            
            # Add param values
            if data_source in data_map and curve_val in data_map[data_source]:
                data_entry = data_map[data_source][curve_val]
                for param_name in all_param_names:
                    if param_name in data_entry:
                        row_parts.append(f"{data_entry[param_name]:.4f}")
                    else:
                        row_parts.append("---")
                # Add R²
                if 'r2' in data_entry and data_entry['r2'] is not None:
                    row_parts.append(f"{data_entry['r2']:.3f}")
                else:
                    row_parts.append("---")
            else:
                row_parts.extend(["---"] * (len(all_param_names) + 1))
            
            print(" & ".join(row_parts) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def _print_table_compact(curve_name, fit_x, curve_short, fit_x_short,
                        all_curve_vals, all_param_names, data_sources, data_map):
    """Print compact table version: model sizes in rows, sources as column groups"""
    
    print(f"\n\\begin{{table}}[H]")
    print("\\centering")
    print(f"\\caption{{$L({curve_short},{fit_x_short})$ Fitting Results - Compact Version}}")
    print(f"\\label{{tab:fitting_results_{curve_short}_{fit_x_short}_compact}}")
    
    # Build column spec: Model Size + (params + R²) per data_source
    n_cols_per_source = len(all_param_names) + 1  # params + R²
    n_sources = len(data_sources)
    col_spec = "l" + "c" * (n_cols_per_source * n_sources)
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    
    # Multi-row header
    # First row: Model Size + source names (without "Model" suffix)
    header_row1 = ["\\multirow{2}{*}{\\textbf{Model Size}}"]
    for data_source in data_sources:
        # Use capitalized short form without "Model" suffix
        ds_short = data_source.capitalize() if data_source in ['base', 'instruct'] else data_source
        header_row1.append(f"\\multicolumn{{{n_cols_per_source}}}{{c}}{{\\textbf{{{ds_short}}}}}")
    print(" & ".join(header_row1) + " \\\\")
    
    # Add cmidrules between source groups
    cmidrules = []
    for i, _ in enumerate(data_sources):
        start_col = 2 + i * n_cols_per_source
        end_col = start_col + n_cols_per_source - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
    print(" ".join(cmidrules))
    
    # Second row: empty + param names per source
    header_row2 = [""]
    for _ in data_sources:
        for param_name in all_param_names:
            header_row2.append(f"${param_name}_{{{fit_x_short}}}$")
        header_row2.append(f"$R^2_{{{fit_x_short}}}$")
    print(" & ".join(header_row2) + " \\\\")
    print("\\midrule")
    
    # Data rows: one row per curve_val
    for curve_val in all_curve_vals:
        # Format curve value
        if curve_val >= 1e6:
            curve_str = (f"{curve_val/1e9:.1f}B" if curve_val/1e9 != int(curve_val/1e9) 
                       else f"{int(curve_val/1e9)}B")
        else:
            curve_str = f"{curve_val:.0f}"
        
        row_parts = [curve_str]
        
        # Add data for each source
        for data_source in data_sources:
            if data_source in data_map and curve_val in data_map[data_source]:
                data_entry = data_map[data_source][curve_val]
                for param_name in all_param_names:
                    if param_name in data_entry:
                        row_parts.append(f"{data_entry[param_name]:.4f}")
                    else:
                        row_parts.append("---")
                # Add R²
                if 'r2' in data_entry and data_entry['r2'] is not None:
                    row_parts.append(f"{data_entry['r2']:.3f}")
                else:
                    row_parts.append("---")
            else:
                row_parts.extend(["---"] * n_cols_per_source)
        
        print(" & ".join(row_parts) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def plot_fit_params_compare_x_combined(args, fitters):
    """Compare-x-combined schema: L(N,C) vs L(N,D) parameter comparison
    
    Creates separate figures for each data_source, with subplots comparing
    L(N,C) and L(N,D) parameters side by side.
    """
    # Group fitters by (data_source, fit_curve)
    fitter_groups = {}
    for fitter in fitters:
        context = fitter.get_context()
        key = (context['data_source'], context['fit_curve'])
        if key not in fitter_groups:
            fitter_groups[key] = []
        fitter_groups[key].append(fitter)
    
    # Process each data_source separately
    data_source_groups = {}
    for (data_source, curve_name), group_fitters in fitter_groups.items():
        if data_source not in data_source_groups:
            data_source_groups[data_source] = []
        data_source_groups[data_source].extend(group_fitters)
    
    for data_source, all_fitters in data_source_groups.items():
        print(f"\n--- Compare-x-combined: {data_source} ---")
        
        # Separate fitters by fit_x
        fit_x_groups = {}
        for fitter in all_fitters:
            context = fitter.get_context()
            fit_x = context['fit_x']
            if fit_x not in fit_x_groups:
                fit_x_groups[fit_x] = []
            fit_x_groups[fit_x].append(fitter)
        
        if len(fit_x_groups) < 2:
            print(f"Warning: Only {len(fit_x_groups)} fit_x found for {data_source}, need at least 2 for comparison")
            continue
        
        # Collect all param names
        all_param_names = set()
        for fitter in all_fitters:
            lookup_params = fitter.get_lookup_params()
            if lookup_params:
                all_param_names.update(lookup_params.keys())
        
        if not all_param_names:
            print(f"No parameters found for {data_source}")
            continue
        
        params_to_plot = _sort_param_names(all_param_names)
        n_params = len(params_to_plot)
        
        # Create figure with subplots: one subplot per parameter
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5), dpi=300)
        if n_params == 1:
            axes = [axes]
        
        # Get curve info (should be same for all fitters)
        first_context = all_fitters[0].get_context()
        curve_name = first_context['fit_curve']
        curve_label = config.DEFAULT_LABELS.get(curve_name, curve_name)
        curve_short = config.DEFAULT_SHORT_NAME.get(curve_name, curve_name)
        
        # Plot each parameter
        for ax_idx, param_name in enumerate(params_to_plot):
            ax = axes[ax_idx]
            
            # Plot each fit_x as different curves
            for fit_x, fit_x_fitters in fit_x_groups.items():
                for fitter in fit_x_fitters:
                    lookup_params = fitter.get_lookup_params()
                    if not lookup_params or param_name not in lookup_params:
                        continue
                    
                    param_table = lookup_params[param_name]
                    x = np.array(sorted(param_table.keys()))
                    y = np.array([param_table[k] for k in x])
                    
                    # Use different colors for different fit_x
                    fit_x_short = config.DEFAULT_SHORT_NAME.get(fit_x, fit_x)
                    color = config.get_color_for_curve(fit_x)
                    marker = config.DEFAULT_MARKERS.get(fit_x, 'o')
                    
                    # Plot with both scatter and line
                    plot.plot_basic(
                        x=x,
                        y=y,
                        use_scatter=True,
                        scatter_alpha=args.scatter_alpha,
                        scatter_s=args.scatter_size,
                        scatter_marker=marker,
                        color=color,
                        ax=ax
                    )
                    
                    # Add legend entry
                    ax.scatter([], [], 
                              color=color,
                              marker=marker,
                              s=args.scatter_size,
                              label=f'L({curve_short},{fit_x_short})')
            
            # Apply settings
            plot.plot_basic_settings(
                ax=ax,
                x_scale=args.plot_x_scale,
                y_scale=args.plot_y_scale,
                x_label=f'{curve_label} ({curve_short})',
                y_label=f'${param_name}({curve_short})$',
                use_legend=args.plot_use_legend,
                legend_loc=args.plot_legend_loc,
                x_tick_on_data=True,
                x_tick_spacing=args.x_tick_spacing,
                y_tick_spacing=args.y_tick_spacing,
                x_grid_spacing=args.x_grid_spacing,
                y_grid_spacing=args.y_grid_spacing,
            )
        
        # Add overall title
        data_source_title = data_source.capitalize() if data_source in ['base', 'instruct'] else data_source
        fit_x_list = sorted(fit_x_groups.keys())
        fit_x_short_list = [config.DEFAULT_SHORT_NAME.get(x, x) for x in fit_x_list]
        title = f"{data_source_title} Model: L({curve_short},{fit_x_short_list[0]}) vs L({curve_short},{fit_x_short_list[1]}) Parameter Comparison"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make room for suptitle
        
        # Save
        args.output_base_dir.mkdir(parents=True, exist_ok=True)
        first_eval = all_fitters[0].get_context()['eval']
        eval_file_str = config.TEST_EVALS[first_eval]['file_str']
        fit_x_list_str = "_vs_".join(sorted(fit_x_list))
        params_str = "_".join(params_to_plot)
        
        save_path = args.output_base_dir / (
            f"{args.output_prefix}{data_source}_{eval_file_str}_"
            f"{curve_name}_{fit_x_list_str}_{params_str}_compare_x_combined.pdf"
        )
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        plt.close()


# Schema registry
PLOT_PARAMS_SCHEMA = {
    'single': plot_fit_params_single,
    'compare-source': plot_fit_params_compare_source,
    'compare-x': plot_fit_params_compare_x,
    'compare-x-combined': plot_fit_params_compare_x_combined,
    'table': plot_fit_params_table,
    'table-compact': plot_fit_params_table_compact,
}