# encoding=utf-8

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "plotly_dark"



# TODO 全部改成dash，因为可以使用Tab展示多维度的数据


def plot_factor_stats(factor_stats):
    """因子"""
    pass


# def plot_ic(ic_test) -> Figure:
#     if ic_test.by_group is None:
#         return _plot_ic_no_group(ic_test)
#     else:
#         return _plot_ic_with_group(ic_test)
#
#
# def _plot_ic_with_group(ic_test):
#
#     pass


def plot_ic_test(ic_test) -> Figure:
    """
    未分组的ic分析结果作图
    Parameters
    ----------
    ic_test: ICTestStruct

    Returns
    -------
    Figure
    """
    ic_series = ic_test.ic_series.to_frame(name='RankIC序列')
    ic_series.index.name = '日期'
    ic_series = ic_series.reset_index()

    ic_cum = ic_test.ic_cum().to_frame(name='RankIC累计值')
    ic_cum.index.name = '日期'
    ic_cum = ic_cum.reset_index()

    table_vals = np.round([ic_test.ic_mean(), ic_test.ic_std(), ic_test.ic_ir(), ic_test.ic_ratio()], 4)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        # shared_xaxes=True,
        # vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"secondary_y": True}]])

    if ic_test.by_group is not None:
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.1, 0.35, 0.55],
            subplot_titles=("RankIC统计指标", "RankIC序列", "行业RankIC统计指标 "),
            # shared_xaxes=True,
            vertical_spacing=0.06,
            specs=[[{"type": "table"}],
                   [{"secondary_y": True}],
                   [{"type": "table"}]
                   ])
    fig.add_trace(
        go.Table(
            header=dict(
                values=["RankIC均值", "RankIC标准差", "IC_IR", "RankIC正值比率"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=list(table_vals),
                align="left")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=ic_series["日期"],
            y=ic_series["RankIC序列"],
            name='RankIC序列'
        ),
        row=2, col=1, secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=ic_cum["日期"],
            y=ic_cum["RankIC累计值"],
            mode="lines",
            name="RankIC累计值"
        ),
        row=2, col=1, secondary_y=True
    )

    if ic_test.by_group is not None:

        gp_ic = pd.DataFrame({"RankIC均值": ic_test.ic_mean(by_group=True),
                              "RankIC标准差": ic_test.ic_std(by_group=True),
                              "IC_IR": ic_test.ic_ir(by_group=True),
                              "RankIC正值比率": ic_test.ic_ratio(by_group=True)}).round(4)

        gp_ic.index.name = '行业'
        gp_ic.reset_index(inplace=True)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["行业名称", "RankIC均值", "RankIC标准差", "IC_IR", "RankIC正值比率"],
                    align="left" ,
                    # line_color='darkslategray',
                    # fill_color='royalblue',
                    # font=dict(color='white', size=12),
                ),
                cells=dict(
                    values=list(gp_ic.values.T),
                    align="left",
                    # line_color='darkslategray',
                    # fill=dict(color=['paleturquoise', 'white']),
                    font_size=12,
                    height=25)
            ),
            row=3, col=1
        )

    fig.update_layout(
        height=1100,
        showlegend=True,
        title_text="因子-[{}]-IC测试结果汇总".format(ic_test.factor_name),
        # legend=dict(
        #     x=1,
        #     y=0.8,
        #     traceorder="normal",
        #     font=dict(
        #         family="sans-serif",
        #         size=12,
        #         color="black"
        #     ),
        #     bgcolor="LightSteelBlue",
        #     bordercolor="Black",
        #     borderwidth=2
        # )
    )

    return fig


def plot_regress_test():
    pass


def plot_quantize_test(quantize_test) -> Figure:

    cum_ret = quantize_test.quantile_cum_return
    # cum_ret.index.name = 'date'
    # value_vars = cum_ret.columns.tolist()
    # cum_ret = cum_ret.reset_index()
    # cum_ret = cum_ret.melt(id_vars='date', value_vars=value_vars)

    mean_ret = quantize_test.quantile_mean_return
    stats = quantize_test.return_stats
    stats = np.round(stats, 4)
    stats.index.name = '指标'
    stats = stats.reset_index()

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.25, 0.35, 0.4],
        subplot_titles=("分组平均收益", "分组累积收益", "收益评价指标 "),
        # shared_xaxes=True,
        vertical_spacing=0.06,
        specs=[[{}],
               [{}],
               [{"type": "table"}]
               ])

    for col in cum_ret:
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret[col],
                mode="lines",
                name=col
            ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=mean_ret.index,
            y=mean_ret,
            name='分组平均收益'
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=stats.columns.tolist(),
                align="left",
                # line_color='darkslategray',
                # fill_color='royalblue',
                # font=dict(color='white', size=12),
            ),
            cells=dict(
                values=list(stats.values.T),
                align="left",
                # line_color='darkslategray',
                # fill=dict(color=['paleturquoise', 'white']),
                font_size=12,
                height=25)
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=1200,
        # showlegend=True,
        title_text="因子-[{}]-分组测试结果汇总".format(quantize_test.factor_name),
        # legend=dict(
        #     x=1,
        #     y=0.8,
        #     traceorder="normal",
        #     font=dict(
        #         family="sans-serif",
        #         size=12,
        #         color="black"
        #     ),
        #     bgcolor="LightSteelBlue",
        #     bordercolor="Black",
        #     borderwidth=2
        # )
    )

    return fig