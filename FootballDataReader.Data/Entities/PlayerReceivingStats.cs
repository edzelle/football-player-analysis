using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Data.Entities
{
    [Table("player_receiving_stats", Schema = "football")]
    public class PlayerReceivingStats
    {
        [Column("year")]
        public DateTime Year { get; set; }

        [Column("player_id")]
        public int PlayerId { get; set; }

        [Column("targets")]
        public int? Targets { get; set; }

        [Column("receptions")]
        public int? Receptions { get; set; }

        [Column("catch_per")]
        public decimal? CatchPercentage { get; set; }

        [Column("yards")]
        public int? Yards { get; set; }

        [Column("yards_per_rec")]
        public decimal? YardsPerReception { get; set; }

        [Column("tds")]
        public int? TDs { get; set; }

        [Column("first_downs_receiving")]
        public int? FirstDownsReceiving { get; set; }

        [Column("receiving_success_rate")]
        public decimal? ReceivingSuccessRate { get; set; }

        [Column("longest_reception")]
        public int? LongestReception { get; set; }

        [Column("yards_per_target")]
        public decimal? YardsPerTarget { get; set; }

        [Column("receptions_per_game")]
        public decimal? ReceptionsPerGame { get; set; }

        [Column("yards_per_game")]
        public decimal? YardsPerGame { get; set; }

        [Column("fumbles")]
        public int? Fumbles { get; set; }

        [Column("overall_usage")]
        public decimal? OverallUseage { get; set; }

        [Column("pass_usage")]
        public decimal? PassUseage { get; set; }

        [Column("rush_usage")]
        public decimal? RushUseage { get; set; }

        [Column("first_down_usage")]
        public decimal? FirstDownUseage { get; set; }

        [Column("second_down_usage")]
        public decimal? SecondDownUseage { get; set; }

        [Column("third_down_usage")]
        public decimal? ThirdDownUseage { get; set; }

        [Column("standard_downs_usage")]
        public decimal? StandardDownsUseage { get; set; }

        [Column("passing_downs_usage")]
        public decimal? PassingDownsUseage { get; set; }

        [Column("average_ppa_all")]
        public decimal? AveragePPAAll { get; set; }

        [Column("average_ppa_pass")]
        public decimal? AveragePPAPass { get; set; }

        [Column("average_ppa_rush")]
        public decimal? AveragePPARush { get; set; }

        [Column("average_ppa_first_down")]
        public decimal? AveragePPAFirstDown { get; set; }

        [Column("average_ppa_second_down")]
        public decimal? AveragePPASecondDown { get; set; }

        [Column("average_ppa_third_down")]
        public decimal? AveragePPAThirdDown { get; set; }

        [Column("average_ppa_standard_downs")]
        public decimal? AveragePPAStandardDowns { get; set; }

        [Column("average_ppa_passing_down")]
        public decimal? AveragePPAPassingDowns { get; set; }

        [Column("total_ppa_all")]
        public decimal? TotalPPAAll { get; set; }

        [Column("total_ppa_pass")]
        public decimal? TotalPPAPass { get; set; }

        [Column("total_ppa_rush")]
        public decimal? TotalPPARush { get; set; }

        [Column("total_ppa_first_down")]
        public decimal? TotalPPAFirstDown { get; set; }

        [Column("total_ppa_second_down")]
        public decimal? TotalPPASecondDown { get; set; }

        [Column("total_ppa_third_down")]
        public decimal? TotalPPAThirdDown { get; set; }

        [Column("total_ppa_standard_downs")]
        public decimal? TotalPPAStandardDowns { get; set; }

        [Column("total_ppa_passing_down")]
        public decimal? TotalPPAPassingDowns { get; set; }

        [ForeignKey("PlayerId")]
        public virtual Player Player { get; set; }
    }
}
